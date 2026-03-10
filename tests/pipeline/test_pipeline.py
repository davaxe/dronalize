"""Tests for the composable Pipeline infrastructure."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dronalize.core import AgentCategory
from dronalize.core.datatypes import LoaderConfig
from dronalize.pipeline import transforms as transform
from dronalize.pipeline.pipeline import Pipeline, ReduceTransform

if TYPE_CHECKING:
    from collections.abc import Iterable

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


def test_pipeline_then_flat_map_returns_new() -> None:
    """Ensure that calling then_flat_map() returns a completely new Pipeline instance."""
    p1 = Pipeline()
    p2 = p1.then_flat_map(_split_by_id)
    assert len(p1) == 0
    assert len(p2) == 1


def test_pipeline_chaining_multiple_steps() -> None:
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


def test_run_chained_transforms_apply_in_order(simple_lf: pl.LazyFrame) -> None:
    """Ensure chained transforms apply in the exact order they were added."""
    # add 1 first, then multiply by 2 → (x+1)*2
    pipe = Pipeline().then(_add_one_to_x).then(_multiply_x_by_two)
    result = pipe.execute_single(simple_lf).collect()
    expected = simple_lf.with_columns((pl.col("x") + 1) * 2).collect()
    assert_frame_equal(result, expected)


def test_run_fan_out_produces_multiple_outputs(simple_lf: pl.LazyFrame) -> None:
    """Verify that a fan-out transform correctly produces multiple output frames."""
    pipe = Pipeline().then_flat_map(_split_by_id)
    results = list(pipe.execute(simple_lf))
    assert len(results) == 2  # two distinct IDs
    # Each should have 4 rows
    for r in results:
        assert r.collect().shape[0] == 4


def test_run_transform_after_fan_out_applied_to_each(simple_lf: pl.LazyFrame) -> None:
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


def test_run_single_succeeds_for_single_output(simple_lf: pl.LazyFrame) -> None:
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


def test_transform_group_by_yield_splits(simple_lf: pl.LazyFrame) -> None:
    """Ensure group_by_yield accurately splits the frame based on the specified column."""
    fan = transform.group_by_yield("id")
    results = list(fan(simple_lf))
    assert len(results) == 2


def test_transform_group_by_yield_drops_col_by_default(simple_lf: pl.LazyFrame) -> None:
    """Verify group_by_yield drops the original grouping column by default."""
    fan = transform.group_by_yield("id")
    for r in fan(simple_lf):
        assert "id" not in r.collect().columns


def test_transform_group_by_yield_keeps_col(simple_lf: pl.LazyFrame) -> None:
    """Verify group_by_yield retains the grouping column when configured to do so."""
    fan = transform.group_by_yield("id", drop_group_cols=False)
    for r in fan(simple_lf):
        assert "id" in r.collect().columns


# ═══════════════════════════════════════════════════════════════════════════
# Domain-specific transform factories
# ═══════════════════════════════════════════════════════════════════════════


def test_transform_yaw_from_vel_default_columns() -> None:
    """Calculate yaw from velocity data using the default column names."""
    lf = pl.DataFrame({"vx": [1.0, 0.0], "vy": [0.0, 1.0]}).lazy()
    fn = transform.yaw_from_vel()
    result = fn(lf).collect()
    assert "yaw" in result.columns
    yaws = result["yaw"].to_list()
    assert yaws[0] == pytest.approx(0.0)
    assert yaws[1] == pytest.approx(math.pi / 2)


def test_transform_yaw_from_vel_custom_columns() -> None:
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


def test_transform_derivative_custom_rename() -> None:
    """Calculate the first derivative and apply a custom renaming scheme."""
    lf = pl.DataFrame({"x": [0.0, 1.0, 4.0, 9.0]}).lazy()
    fn = transform.derivative("x", dt=1.0, n=1, derivative_rename={1: ["velocity"]})
    result = fn(lf).collect()
    assert "velocity" in result.columns


def test_transform_derivative_second_with_intermediate() -> None:
    """Calculate the second derivative and include the intermediate first derivative."""
    lf = pl.DataFrame({"x": [0.0, 1.0, 4.0, 9.0, 16.0]}).lazy()
    fn = transform.derivative("x", dt=1.0, n=2, include_intermediate=True)
    result = fn(lf).collect()
    assert "d1_x" in result.columns
    assert "d2_x" in result.columns


def test_transform_filter_no_config_passes_all(trajectory_lf: pl.LazyFrame) -> None:
    """Ensure the filter transform passes all rows when no specific rules apply."""
    config = LoaderConfig(input_len=3, output_len=3, sample_time=0.1)
    fn = transform.filter_scene(config.filtering)
    result = fn(trajectory_lf).collect()
    assert_frame_equal(result, trajectory_lf.collect())


def test_transform_filter_removes_category(trajectory_lf: pl.LazyFrame) -> None:
    """Verify the filter transform removes agents that match the specified filtering category."""
    # Filter out agent_category == 1 -> should remove all

    config = LoaderConfig(input_len=3, output_len=3, sample_time=0.1).with_filtering(
        filter_agent_category=[AgentCategory.CAR]
    )
    fn = transform.filter_scene(config.filtering)
    result = fn(trajectory_lf).collect()
    # All agents have category 1 (CAR), and min_agents=2 by default,
    # so after filtering CAR agents out, no valid agents remain,
    # thus the entire frame should be filtered
    assert result.shape[0] == 0


def test_transform_resample_identity(simple_lf: pl.LazyFrame) -> None:
    """Ensure resampling with identical rates accurately preserves the row count."""
    config = LoaderConfig(input_len=2, output_len=2, sample_time=1.0)
    fn = transform.resample(config.resampling, config.sample_time, group_by="id")
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
    config = LoaderConfig(input_len=5, output_len=5, sample_time=0.1).with_resampling(1, 2)
    fn = transform.resample(config.resampling, config.sample_time)
    result = fn(lf).collect()
    assert result.shape[0] == 5


def test_transform_rebalance_basic() -> None:
    """Rebalance the dataset based on lane changes and check the final row counts."""
    lf = pl.DataFrame({
        "id": [0, 1, 2, 3, 4, 5],
        "lane_changes": [0, 0, 0, 0, 2, 2],
        "x": [0.0] * 6,
    }).lazy()
    fn = transform.rebalance(2.0, seed=42)
    result = fn(lf).collect()
    # 2 LC agents + 1 LK agent = 3
    assert result.shape[0] == 3
    # lane_changes column should be dropped by default
    assert "lane_changes" not in result.columns


def test_transform_rebalance_keep_column() -> None:
    """Rebalance the dataset and ensure the lane changes column is retained."""
    lf = pl.DataFrame({
        "id": [0, 1, 2, 3],
        "lane_changes": [0, 0, 2, 2],
        "x": [0.0] * 4,
    }).lazy()
    fn = transform.rebalance(1.0, seed=42, drop_lanechange_col=False)
    result = fn(lf).collect()
    assert "lane_changes" in result.columns


def test_transform_window_produces_multiple() -> None:
    """Ensure the window transform properly produces multiple windowed frames."""
    lf = pl.DataFrame({
        "frame": list(range(10)),
        "x": [float(i) for i in range(10)],
    }).lazy()
    fan = transform.window(window_size=5, step_size=2, return_iterable=True)
    results = list(fan(lf))
    assert len(results) >= 2


def test_transform_window_offsets_frame() -> None:
    """Verify the window transform completely resets the starting frame offset to zero."""
    lf = pl.DataFrame({
        "frame": list(range(10)),
        "x": [float(i) for i in range(10)],
    }).lazy()
    fan = transform.window(window_size=5, step_size=3, return_iterable=True)
    for result_lf in fan(lf):
        result = result_lf.collect()
        # Frame should be zero-offset
        assert result["frame"].min() == 0


def test_transform_window_no_offset() -> None:
    """Verify the window transform retains the original frame offset."""
    lf = pl.DataFrame({
        "frame": list(range(6)),
        "x": [float(i) for i in range(6)],
    }).lazy()
    fan = transform.window(
        window_size=5, step_size=3, offset_sliding_col=False, return_iterable=True
    )
    results = [r.collect() for r in fan(lf)]
    # At least first window should start at frame 0
    assert results[0]["frame"].min() == 0


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline with transforms integration
# ═══════════════════════════════════════════════════════════════════════════


def test_pipeline_integration_filter_then_yaw() -> None:
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


def test_pipeline_integration_window_then_transform() -> None:
    """Apply a windowing fan-out followed immediately by a custom transform on each window."""
    lf = pl.DataFrame({
        "frame": list(range(10)),
        "x": [float(i) for i in range(10)],
        "y": [0.0] * 10,
    }).lazy()
    pipe = (
        Pipeline()
        .then_flat_map(transform.window(window_size=5, step_size=3, return_iterable=True))
        .then(lambda df: df.with_columns(pl.lit(99).alias("marker")))
    )
    results = [r.collect() for r in pipe.execute(lf)]
    assert len(results) >= 2
    for r in results:
        assert "marker" in r.columns
        assert r["marker"].unique().to_list() == [99]


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
# Additional Helpers / fixtures for Reduce
# ═══════════════════════════════════════════════════════════════════════════


def _concat_reduce(dfs: Iterable[pl.LazyFrame]) -> pl.LazyFrame:
    """Reduce by concatination."""
    frames = list(dfs)
    if not frames:
        return pl.LazyFrame()
    return pl.concat(frames)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline construction & immutability (Add to existing section)
# ═══════════════════════════════════════════════════════════════════════════


def test_pipeline_then_reduce_returns_new() -> None:
    """Ensure that calling then_reduce() returns a completely new Pipeline instance."""
    p1 = Pipeline()
    p2 = p1.then_reduce(_concat_reduce)
    assert len(p1) == 0, "Original must be unchanged"
    assert len(p2) == 1


def test_pipeline_repr_reduce() -> None:
    """Check the string representation of a Pipeline containing a reduce step."""
    pipe = Pipeline().then_reduce(_concat_reduce)
    r = repr(pipe)
    assert ".then_reduce(" in r


def test_pipeline_rshift_operator() -> None:
    """Verify the >> operator correctly composes pipelines."""
    p1 = Pipeline().then(_add_one_to_x)
    p2 = Pipeline().then(_multiply_x_by_two)
    combined = p1 >> p2
    assert len(combined) == 2
    # Originals unchanged
    assert len(p1) == 1
    assert len(p2) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline execution (Add to existing section)
# ═══════════════════════════════════════════════════════════════════════════


def test_run_reduce_combines_multiple_streams(simple_lf: pl.LazyFrame) -> None:
    """Verify that a reduce step correctly combines multiple frames into one."""
    pipe = Pipeline().then_flat_map(_split_by_id).then(_add_one_to_x).then_reduce(_concat_reduce)
    result = pipe.execute_single(simple_lf).collect()

    # The expected frame is the original but with x + 1.
    # Because group_by and concat might alter row order, sort before comparing.
    expected = simple_lf.with_columns(pl.col("x") + 1).collect()

    result_sorted = result.sort(["id", "frame"])
    expected_sorted = expected.sort(["id", "frame"])

    assert_frame_equal(result_sorted, expected_sorted)


def test_run_single_succeeds_after_reduce(simple_lf: pl.LazyFrame) -> None:
    """Ensure run_single succeeds seamlessly when a flat-map is followed by a reduce."""
    # A flat-map normally causes execute_single to raise an error.
    # Reducing it back down to a single frame should allow it to pass.
    pipe = Pipeline().then_flat_map(_split_by_id).then_reduce(_concat_reduce)
    result = pipe.execute_single(simple_lf).collect()

    # Original frame had 8 rows, splitting into 2x4 and recombining should equal 8
    assert result.shape[0] == 8
    assert "id" in result.columns


def test_run_reduce_on_empty_stream() -> None:
    """Verify the reduce step handles an empty iterable of LazyFrames gracefully."""
    # Simulate an empty stream by filtering out everything before flat-mapping
    empty_lf = pl.DataFrame({"id": [], "x": []}).lazy()

    pipe = Pipeline().then_flat_map(_split_by_id).then_reduce(_concat_reduce)

    result = pipe.execute_single(empty_lf).collect()
    assert result.shape[0] == 0


def test_run_lazy_reduce(simple_lf: pl.LazyFrame) -> None:
    """Ensure then_lazy_reduce constructs and applies the reduce factory correctly."""

    def reduce_factory() -> ReduceTransform:
        return _concat_reduce

    pipe = Pipeline().then_flat_map(_split_by_id).then_lazy_reduce(reduce_factory)

    result = pipe.execute_single(simple_lf).collect()
    assert result.shape[0] == 8


def test_run_if_present_reduce(simple_lf: pl.LazyFrame) -> None:
    """Verify then_if_present_reduce applies conditionally based on the argument."""

    def reduce_factory(arg: str) -> ReduceTransform:
        # Return a custom reduce that adds a column to indicate the argument was passed
        def custom_reduce(dfs: Iterable[pl.LazyFrame]) -> pl.LazyFrame:
            frames = list(dfs)
            return pl.concat(frames).with_columns(pl.lit(arg).alias("reduce_arg"))

        return custom_reduce

    # Case 1: Arg is provided
    pipe_with = (
        Pipeline().then_flat_map(_split_by_id).then_if_present_reduce(reduce_factory, "test_arg")
    )
    result_with = pipe_with.execute_single(simple_lf).collect()
    assert "reduce_arg" in result_with.columns

    pipe_without = (
        Pipeline().then_flat_map(_split_by_id).then_if_present_reduce(reduce_factory, None)
    )

    with pytest.raises(ValueError, match="exactly 1 output"):
        pipe_without.execute_single(simple_lf)
