# pyright: standard
"""Tests for the composable Pipeline infrastructure."""

from __future__ import annotations

import math
from collections import Counter

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dronalize.core.categories import AgentCategory
from dronalize.processing.filters import Filter, cleanup
from dronalize.processing.ingest import (
    LoaderConfig,
    ShuffledTimeBlockSplit,
    SplitRequest,
    SplitWeights,
    TimeBlockSplit,
)
from dronalize.processing.pipeline import Pipeline
from dronalize.processing.pipeline import transforms as transform
from dronalize.processing.pipeline.extensions import LaneChangeSamplingExtension
from dronalize.processing.pipeline.factory import trajectory_pipeline
from dronalize.processing.pipeline.functional.resample import ResampleSpec
from dronalize.processing.pipeline.presets import highway_trajectory_spec, standard_trajectory_spec
from dronalize.processing.pipeline.spec import LaneChangeDetection
from dronalize.processing.pipeline.splitting import split_partition_pipeline

# ═══════════════════════════════════════════════════════════════════════════
# Helpers / fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def simple_lf() -> pl.LazyFrame:
    """Return a tiny LazyFrame with frame, id, x, and y columns."""
    return pl.DataFrame({
        "frame": [0, 1, 2, 3, 0, 1, 2, 3],
        "id": [1, 1, 1, 1, 2, 2, 2, 2],
        "x": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
        "y": [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0],
    }).lazy()


@pytest.fixture
def trajectory_lf() -> pl.LazyFrame:
    """Return a LazyFrame with trajectory data including velocity and category."""
    return pl.DataFrame({
        "frame": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        "id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "vx": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "vy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "agent_category": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }).lazy()


def _add_one_to_x(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(pl.col("x") + 1)


def _multiply_x_by_two(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(pl.col("x") * 2)


def _split_by_id(df: pl.LazyFrame) -> list[pl.LazyFrame]:
    collected = df.collect()
    return [group.lazy() for _, group in collected.group_by("id")]


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline construction & immutability
# ═══════════════════════════════════════════════════════════════════════════


def test_pipeline_empty() -> None:
    """Verify that a newly instantiated pipeline is empty and falsy."""
    pipe = Pipeline()
    assert len(pipe) == 0
    assert not pipe  # falsy


def test_pipeline_then_returns_new() -> None:
    """Ensure that calling then() returns a completely new Pipeline instance."""
    p1 = Pipeline()
    p2 = p1.then(_add_one_to_x)
    assert len(p1) == 0, "Original must be unchanged"
    assert len(p2) == 1


def test_then_flat_map_is_immutable() -> None:
    """Ensure that calling then_flat_map() returns a completely new Pipeline instance."""
    p1 = Pipeline()
    p2 = p1.then_flat_map(_split_by_id)
    assert len(p1) == 0
    assert len(p2) == 1


def test_pipeline_chaining() -> None:
    """Verify that chaining multiple steps works and evaluates as truthy."""
    pipe = Pipeline().then(_add_one_to_x).then(_multiply_x_by_two)
    assert len(pipe) == 2
    assert pipe  # truthy


def test_pipeline_compose_combines_two() -> None:
    """Verify that composing two pipelines combines their steps without modifying the originals."""
    p1 = Pipeline().then(_add_one_to_x)
    p2 = Pipeline().then(_multiply_x_by_two)
    combined = p1.compose(p2)
    assert len(combined) == 2
    # Originals unchanged
    assert len(p1) == 1
    assert len(p2) == 1


def test_pipeline_repr_non_empty() -> None:
    """Check the string representation of a non-empty Pipeline."""
    pipe = Pipeline().then(_add_one_to_x).then_flat_map(_split_by_id)
    r = repr(pipe)
    assert "Pipeline" in r
    assert ".then(" in r
    assert ".then_flat_map(" in r


def test_pipeline_repr_empty() -> None:
    """Check the string representation of an empty Pipeline."""
    r = repr(Pipeline())
    assert "empty" in r


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline execution
# ═══════════════════════════════════════════════════════════════════════════


def test_run_empty_pipeline_is_identity(simple_lf: pl.LazyFrame) -> None:
    """Ensure running an empty pipeline acts as an identity function."""
    results = list(Pipeline().execute(simple_lf))
    assert len(results) == 1
    assert_frame_equal(results[0].collect(), simple_lf.collect())


def test_run_single_transform(simple_lf: pl.LazyFrame) -> None:
    """Verify that a single transform applies correctly to a LazyFrame."""
    pipe = Pipeline().then(_add_one_to_x)
    result = pipe.execute_single(simple_lf).collect()
    expected = simple_lf.with_columns(pl.col("x") + 1).collect()
    assert_frame_equal(result, expected)


def test_run_chained_transforms(simple_lf: pl.LazyFrame) -> None:
    """Ensure chained transforms apply in the exact order they were added."""
    # add 1 first, then multiply by 2 → (x+1)*2
    pipe = Pipeline().then(_add_one_to_x).then(_multiply_x_by_two)
    result = pipe.execute_single(simple_lf).collect()
    expected = simple_lf.with_columns((pl.col("x") + 1) * 2).collect()
    assert_frame_equal(result, expected)


def test_run_fan_out(simple_lf: pl.LazyFrame) -> None:
    """Verify that a fan-out transform correctly produces multiple output frames."""
    pipe = Pipeline().then_flat_map(_split_by_id)
    results = list(pipe.execute(simple_lf))
    assert len(results) == 2  # two distinct IDs
    # Each should have 4 rows
    for r in results:
        assert r.collect().shape[0] == 4


def test_run_after_fan_out(simple_lf: pl.LazyFrame) -> None:
    """Ensure transforms added after a fan-out apply independently to all resulting frames."""
    pipe = Pipeline().then_flat_map(_split_by_id).then(_add_one_to_x)
    results = [r.collect() for r in pipe.execute(simple_lf)]
    assert len(results) == 2
    # All x values should have been incremented by 1
    all_x = pl.concat(results)["x"].to_list()
    original_x = simple_lf.collect()["x"].to_list()
    for orig, new in zip(sorted(original_x), sorted(all_x), strict=True):
        assert new == pytest.approx(orig + 1)


def test_run_single_raises_on_fan_out(simple_lf: pl.LazyFrame) -> None:
    """Verify that run_single raises a ValueError when multiple frames are produced."""
    pipe = Pipeline().then_flat_map(_split_by_id)
    with pytest.raises(ValueError, match="exactly 1 output"):
        pipe.execute_single(simple_lf)


def test_run_single_one_output(simple_lf: pl.LazyFrame) -> None:
    """Ensure run_single succeeds seamlessly when exactly one frame is produced."""
    pipe = Pipeline().then(_add_one_to_x)
    result = pipe.execute_single(simple_lf)
    assert isinstance(result, pl.LazyFrame)


def test_compose_then_run(simple_lf: pl.LazyFrame) -> None:
    """Verify that a composed pipeline runs all internal transforms correctly."""
    p1 = Pipeline().then(_add_one_to_x)
    p2 = Pipeline().then(_multiply_x_by_two)
    combined = p1.compose(p2)
    result = combined.execute_single(simple_lf).collect()
    expected = simple_lf.with_columns((pl.col("x") + 1) * 2).collect()
    assert_frame_equal(result, expected)


def test_lambda_transform(simple_lf: pl.LazyFrame) -> None:
    """Ensure lambda functions correctly mutate the data within the pipeline."""
    pipe = Pipeline().then(lambda df: df.with_columns(pl.col("x") + 100))
    result = pipe.execute_single(simple_lf).collect()
    min_x = result["x"].min()
    assert min_x is not None
    assert isinstance(min_x, (int, float))
    assert min_x >= 100


# ═══════════════════════════════════════════════════════════════════════════
# Built-in transforms
# ═══════════════════════════════════════════════════════════════════════════


def test_lambda_with_columns(simple_lf: pl.LazyFrame) -> None:
    """Verify a lambda transform using with_columns adds the expected columns."""
    pipe = Pipeline().then(lambda df: df.with_columns(pl.lit(42).alias("answer")))
    result = pipe.execute_single(simple_lf).collect()
    assert "answer" in result.columns
    assert result["answer"].unique().to_list() == [42]


def test_lambda_select(simple_lf: pl.LazyFrame) -> None:
    """Verify a lambda transform using select keeps only the specified columns."""
    pipe = Pipeline().then(lambda df: df.select("frame", "id"))
    result = pipe.execute_single(simple_lf).collect()
    assert result.columns == ["frame", "id"]


def test_lambda_drop(simple_lf: pl.LazyFrame) -> None:
    """Verify a lambda transform using drop successfully removes specified columns."""
    pipe = Pipeline().then(lambda df: df.drop("y"))
    result = pipe.execute_single(simple_lf).collect()
    assert "y" not in result.columns
    assert "x" in result.columns


def test_lambda_rename(simple_lf: pl.LazyFrame) -> None:
    """Verify a lambda transform using rename alters the column names correctly."""
    pipe = Pipeline().then(lambda df: df.rename({"x": "pos_x", "y": "pos_y"}))
    result = pipe.execute_single(simple_lf).collect()
    assert "pos_x" in result.columns
    assert "pos_y" in result.columns
    assert "x" not in result.columns


def test_lambda_filter(simple_lf: pl.LazyFrame) -> None:
    """Verify a lambda transform using filter retains only the correct rows."""
    pipe = Pipeline().then(lambda df: df.filter(pl.col("id") == 1))
    result = pipe.execute_single(simple_lf).collect()
    assert result["id"].unique().to_list() == [1]
    assert result.shape[0] == 4


def test_lambda_sort() -> None:
    """Verify a lambda transform using sort properly orders the rows."""
    lf = pl.DataFrame({"a": [3, 1, 2]}).lazy()
    pipe = Pipeline().then(lambda df: df.sort("a"))
    assert pipe.execute_single(lf).collect()["a"].to_list() == [1, 2, 3]


def test_group_by_yield_splits(simple_lf: pl.LazyFrame) -> None:
    """Ensure group_by_yield accurately splits the frame based on the specified column."""
    fan = transform.group_by_yield("id")
    results = list(fan(simple_lf))
    assert len(results) == 2


def test_group_by_yield_drops_group_col(simple_lf: pl.LazyFrame) -> None:
    """Verify group_by_yield drops the original grouping column by default."""
    fan = transform.group_by_yield("id")
    for r in fan(simple_lf):
        assert "id" not in r.collect().columns


def test_group_by_yield_keeps_group_col(simple_lf: pl.LazyFrame) -> None:
    """Verify group_by_yield retains the grouping column when configured to do so."""
    fan = transform.group_by_yield("id", drop_group_cols=False)
    for r in fan(simple_lf):
        assert "id" in r.collect().columns


# ═══════════════════════════════════════════════════════════════════════════
# Domain-specific transform factories
# ═══════════════════════════════════════════════════════════════════════════


def test_transform_yaw_from_vel() -> None:
    """Calculate yaw from velocity data using the default column names."""
    lf = pl.DataFrame({"vx": [1.0, 0.0], "vy": [0.0, 1.0]}).lazy()
    fn = transform.yaw_from_vel()
    result = fn(lf).collect()
    assert "yaw" in result.columns
    yaws = result["yaw"].to_list()
    assert yaws[0] == pytest.approx(0.0)
    assert yaws[1] == pytest.approx(math.pi / 2)


def test_transform_yaw_from_vel_custom() -> None:
    """Calculate yaw from velocity data using custom column names."""
    lf = pl.DataFrame({"vel_x": [1.0], "vel_y": [0.0]}).lazy()
    fn = transform.yaw_from_vel(vx_col="vel_x", vy_col="vel_y", yaw_col="heading")
    result = fn(lf).collect()
    assert "heading" in result.columns


def test_transform_yaw_from_pos_basic() -> None:
    """Calculate yaw accurately based on position deltas."""
    lf = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]}).lazy()
    fn = transform.yaw_from_pos()
    result = fn(lf).collect()
    assert "yaw" in result.columns
    # Second row: dx=1, dy=0 → yaw=0
    assert result["yaw"][1] == pytest.approx(0.0)


def test_transform_derivative_first() -> None:
    """Calculate the first derivative for a specified column over time."""
    lf = pl.DataFrame({"x": [0.0, 1.0, 4.0, 9.0]}).lazy()
    fn = transform.derivative("x", dt=1.0, n=1)
    result = fn(lf).collect()
    assert "d1_x" in result.columns


def test_transform_derivative_rename() -> None:
    """Calculate the first derivative and apply a custom renaming scheme."""
    lf = pl.DataFrame({"x": [0.0, 1.0, 4.0, 9.0]}).lazy()
    fn = transform.derivative("x", dt=1.0, n=1, derivative_rename={1: ["velocity"]})
    result = fn(lf).collect()
    assert "velocity" in result.columns


def test_transform_second_derivative() -> None:
    """Calculate the second derivative and include the intermediate first derivative."""
    lf = pl.DataFrame({"x": [0.0, 1.0, 4.0, 9.0, 16.0]}).lazy()
    fn = transform.derivative("x", dt=1.0, n=2, include_intermediate=True)
    result = fn(lf).collect()
    assert "d1_x" in result.columns
    assert "d2_x" in result.columns


def test_transform_filter_category(trajectory_lf: pl.LazyFrame) -> None:
    """Verify the filter transform removes agents that match the specified filtering category."""
    # Filter out agent_category == 1 -> should remove all

    config = LoaderConfig(input_len=3, output_len=3, sample_time=0.1).with_filter(
        Filter.define(
            cleanup_rules=[cleanup.ExcludeCategories.define(categories=[AgentCategory.CAR])]
        )
    )
    fn = transform.filter_scene(config.filter)
    result = fn(trajectory_lf).collect()
    # All agents have category 1 (CAR), so cleanup removes every row.
    assert result.shape[0] == 0


def test_transform_resample_identity(simple_lf: pl.LazyFrame) -> None:
    """Ensure resampling with identical rates accurately preserves the row count."""
    config = LoaderConfig(input_len=2, output_len=2, sample_time=1.0)
    fn = transform.resample(config.resampling, group_by="id")
    result = fn(simple_lf).collect()
    # 1:1 resampling should preserve row count
    assert result.shape[0] == simple_lf.collect().shape[0]


def test_transform_resample_downsample() -> None:
    """Verify the resampling transform correctly downsamples the frame."""
    lf = pl.DataFrame({
        "frame": list(range(10)),
        "x": [float(i) for i in range(10)],
        "y": [0.0] * 10,
    }).lazy()
    config = LoaderConfig(input_len=5, output_len=5, sample_time=0.1).with_resampling(
        ResampleSpec(up=1, down=2)
    )
    fn = transform.resample(spec=config.resampling)
    result = fn(lf).collect()
    assert result.shape[0] == 5


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline with transforms integration
# ═══════════════════════════════════════════════════════════════════════════


def test_pipeline_filter_then_yaw() -> None:
    """Apply a filter and then calculate yaw to verify transform integration."""
    lf = pl.DataFrame({
        "id": [1, 1, 2, 2],
        "vx": [1.0, 1.0, -1.0, -1.0],
        "vy": [0.0, 0.0, 0.0, 0.0],
    }).lazy()
    pipe = Pipeline().then(lambda df: df.filter(pl.col("id") == 1)).then(transform.yaw_from_vel())
    result = pipe.execute_single(lf).collect()
    assert result.shape[0] == 2
    assert result["yaw"][0] == pytest.approx(0.0)


def test_pipeline_integration_complex() -> None:
    """Execute a complex pipeline with filtering, mapping, fan-out, and renaming."""
    lf = pl.DataFrame({
        "frame": [0, 1, 2, 3, 0, 1, 2, 3],
        "id": [1, 1, 1, 1, 2, 2, 2, 2],
        "x": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
        "y": [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0],
    }).lazy()
    pipe = (
        Pipeline()
        .then(lambda df: df.with_columns(pl.col("x") + 100))
        .then_flat_map(transform.group_by_yield("id", drop_group_cols=False))
        .then(lambda df: df.rename({"x": "pos_x"}))
    )
    results = [r.collect() for r in pipe.execute(lf)]
    assert len(results) == 2
    for r in results:
        assert "pos_x" in r.columns
        min_x = r["pos_x"].min()
        assert min_x is not None
        assert isinstance(min_x, (float, int))
        assert min_x >= 100.0


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline construction & immutability (Add to existing section)
# ═══════════════════════════════════════════════════════════════════════════


def test_pipeline_rshift_operator() -> None:
    """Verify the >> operator correctly composes pipelines."""
    p1 = Pipeline().then(_add_one_to_x)
    p2 = Pipeline().then(_multiply_x_by_two)
    combined = p1 >> p2
    assert len(combined) == 2
    # Originals unchanged
    assert len(p1) == 1
    assert len(p2) == 1


# =══════════════════════════════════════════════════════════════════════════
# Split partitioning pipeline
# ══════════════════════════════════════════════════════════════════════════=


def test_block_partition_cumulative() -> None:
    """Test cumulative block partitioning."""
    lf = pl.DataFrame({
        "frame": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "value": [11, 12, 13, 14, 20, 21, 22, 23, 24, 25],
    }).lazy()

    split_request = SplitRequest(
        strategy=TimeBlockSplit(gap=0), weights=SplitWeights.from_tuple((0.6, 0.2, 0.2))
    )
    fn = split_partition_pipeline(request=split_request, time_column="frame")
    result = fn.execute_single(lf).collect()
    assert "split" in result.columns
    splits = result["split"]
    assert splits.n_unique() == 3
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    splits_list = splits.to_list()
    assert splits_list[:6] == ["train"] * 6
    assert splits_list[6:8] == ["val"] * 2
    assert splits_list[8:] == ["test"] * 2


def test_block_partition_cumulative_gap() -> None:
    """Test that gap and only two splits works."""
    lf = pl.DataFrame({
        "frame": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "value": [11, 12, 13, 14, 20, 21, 22, 23, 24, 25],
    }).lazy()

    split_request = SplitRequest(
        strategy=TimeBlockSplit(gap=2), weights=SplitWeights.from_tuple((0.5, 0.5, 0.0))
    )
    fn = split_partition_pipeline(request=split_request, time_column="frame")
    result = fn.execute_single(lf).collect()
    assert "split" in result.columns
    assert result.height == 8
    splits = result["split"]
    assert splits.n_unique() == 2
    assert "train" in splits
    assert "val" in splits
    splits_list = splits.to_list()
    # 2 gap means that 8 frames remains; first 4 should be train
    assert splits_list[:4] == ["train"] * 4
    # 2 gap is removed from the middle, so last 4 should be val
    assert splits_list[4:] == ["val"] * 4


def test_block_partition_shuffled() -> None:
    """Test shuffled block partitioning."""
    lf = pl.DataFrame({
        "frame": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "value": [11, 12, 13, 14, 20, 21, 22, 23, 24, 25],
    }).lazy()

    split_request = SplitRequest(
        strategy=ShuffledTimeBlockSplit(segments=5),
        weights=SplitWeights.from_tuple((0.6, 0.2, 0.2)),
        seed=0,
    )
    fn = split_partition_pipeline(request=split_request, time_column="frame")
    result = fn.execute_single(lf).collect()
    assert "split" in result.columns
    splits = result["split"]
    assert splits.n_unique() == 3
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    counts = Counter(splits.to_list())
    assert counts["train"] == 6
    assert counts["val"] == 2
    assert counts["test"] == 2
    # fmt: off
    # The order is shuffled by segment, but with seed 0 it is deterministic.
    assert splits.to_list() == [
        "train", "train", "val", "val", "test",
        "test", "train", "train", "train", "train"
    ]
    # fmt: on


def test_block_partition_shuffled_gap() -> None:
    """Test that gap and only two splits works for shuffled block partitioning."""
    lf = pl.DataFrame({
        "frame": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "id": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "value": [11, 12, 13, 14, 20, 21, 22, 23, 24, 25],
    }).lazy()

    split_request = SplitRequest(
        strategy=ShuffledTimeBlockSplit(segments=2, gap=2),
        weights=SplitWeights.from_tuple((0.5, 0.5, 0.0)),
        seed=42,
    )
    fn = split_partition_pipeline(request=split_request, time_column="frame")
    result = fn.execute_single(lf).collect()
    print(result)
    assert "split" in result.columns
    assert result.height == 8
    splits = result["split"]
    assert splits.n_unique() == 2
    assert "train" in splits
    assert "val" in splits

    counts = Counter(splits.to_list())
    assert counts["train"] == 4
    assert counts["val"] == 4


def test_pipeline_keeps_shuffled_segments() -> None:
    """Shuffled time-block splits should fan out one scene per contiguous segment."""
    lf = pl.DataFrame({
        "frame": list(range(8)),
        "id": [1] * 8,
        "x": [float(frame) for frame in range(8)],
        "y": [0.0] * 8,
    }).lazy()
    pipeline = trajectory_pipeline(
        standard_trajectory_spec(
            LoaderConfig(input_len=1, output_len=1, sample_time=1.0),
            split_request=SplitRequest(
                strategy=ShuffledTimeBlockSplit(segments=4),
                weights=SplitWeights.from_tuple((0.5, 0.5, 0.0)),
                seed=0,
            ),
        )
    )

    results = list(pipeline.execute(lf, collect=True))

    assert len(results) == 4
    assert [result.height for result in results] == [2, 2, 2, 2]
    assert all(result["split"].n_unique() == 1 for result in results)


def test_pipeline_windows_within_segments() -> None:
    """Windowing should stay inside each shuffled time-block segment instead of merging by split."""
    lf = pl.DataFrame({
        "frame": list(range(8)),
        "id": [1] * 8,
        "x": [float(frame) for frame in range(8)],
        "y": [0.0] * 8,
    }).lazy()
    pipeline = trajectory_pipeline(
        standard_trajectory_spec(
            LoaderConfig(input_len=1, output_len=1, sample_time=1.0).with_window(step_size=2),
            split_request=SplitRequest(
                strategy=ShuffledTimeBlockSplit(segments=4),
                weights=SplitWeights.from_tuple((0.5, 0.5, 0.0)),
                seed=0,
            ),
        )
    )

    results = list(pipeline.execute(lf, collect=True))

    assert len(results) == 4
    assert [result.height for result in results] == [2, 2, 2, 2]
    assert all(result["split"].n_unique() == 1 for result in results)


def test_highway_spec_lane_change_defaults() -> None:
    """The highway spec helper should configure lane-aware sampling succinctly."""
    spec = highway_trajectory_spec(
        LoaderConfig(input_len=1, output_len=1, sample_time=1.0),
        negative_keep_every=4,
        min_lane_change_events=2,
        lane_change=LaneChangeDetection(persist=2, margin_before=1),
    )

    assert spec.columns.lane_id == "lane_id"
    assert spec.extension == LaneChangeSamplingExtension(
        negative_keep_every=4, min_lane_change_events=2, persist=2, margin_before=1, margin_after=0
    )


def test_standard_spec_policy() -> None:
    """The standard spec helper should not attach an extension by default."""
    spec = standard_trajectory_spec(LoaderConfig(input_len=1, output_len=1, sample_time=1.0))

    assert spec.extension is None


def test_window_by_keeps_groups() -> None:
    """Scene grouping should still fan out when only window_by is configured."""
    lf = pl.DataFrame({
        "recording": [1, 1, 2, 2],
        "frame": [0, 1, 0, 1],
        "id": [1, 1, 1, 1],
        "x": [0.0, 1.0, 10.0, 11.0],
        "y": [0.0, 0.0, 0.0, 0.0],
    }).lazy()

    pipeline = trajectory_pipeline(
        standard_trajectory_spec(
            LoaderConfig(input_len=1, output_len=1, sample_time=1.0), window_by="recording"
        )
    )

    results = list(pipeline.execute(lf, collect=True))

    assert len(results) == 2
    assert sorted(result["recording"].unique().item() for result in results) == [1, 2]


def test_highway_sampling_thins_negatives() -> None:
    """Lane-change windows should be kept while no-change windows are downsampled."""
    lf = pl.DataFrame({
        "frame": list(range(8)),
        "id": [1] * 8,
        "x": [float(frame) for frame in range(8)],
        "y": [0.0] * 8,
        "lane_id": [1, 1, 1, 2, 2, 2, 2, 2],
    }).lazy()
    pipeline = trajectory_pipeline(
        highway_trajectory_spec(
            LoaderConfig(input_len=1, output_len=2, sample_time=1.0).with_window(step_size=1),
            negative_keep_every=2,
        )
    )

    results = list(pipeline.execute(lf, collect=True))
    starts = [result["x"][0] for result in results]

    assert starts == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_highway_sampling_requires_multiple_changes() -> None:
    """A window should only count as positive after the configured event threshold."""
    lf = pl.DataFrame({
        "frame": list(range(10)),
        "id": [1] * 10,
        "x": [float(frame) for frame in range(10)],
        "y": [0.0] * 10,
        "lane_id": [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
    }).lazy()
    pipeline = trajectory_pipeline(
        highway_trajectory_spec(
            LoaderConfig(input_len=1, output_len=3, sample_time=1.0).with_window(step_size=1),
            negative_keep_every=10,
            min_lane_change_events=2,
        )
    )

    results = list(pipeline.execute(lf, collect=True))
    starts = [result["x"][0] for result in results]

    assert starts == [0.0, 2.0]
