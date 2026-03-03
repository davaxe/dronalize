"""Tests for the composable Pipeline infrastructure."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from typing_extensions import override

from dronalize.core import AgentCategory
from dronalize.core.datatypes import LoaderConfig
from dronalize.core.datatypes import map_context as mc
from dronalize.core.pipeline import (
    FlatMapTransform,
    Pipeline,
    Transform,
    pipeline_from_config,
)
from dronalize.core.pipeline import transforms as transform
from dronalize.core.protocols.loader import BaseSceneLoader, Source

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
# Protocol conformance
# ═══════════════════════════════════════════════════════════════════════════


def test_plain_function_is_transform() -> None:
    """Verify that regular functions satisfy the Transform protocol."""
    assert isinstance(_add_one_to_x, Transform)


def test_lambda_is_transform() -> None:
    """Verify that lambda functions satisfy the Transform protocol."""
    fn = lambda df: df.with_columns(pl.col("x") + 1)  # noqa: E731
    assert isinstance(fn, Transform)


def test_fan_out_function_is_fanout() -> None:
    """Verify that regular fan-out functions satisfy the FlatMapTransform protocol."""
    assert isinstance(_split_by_id, FlatMapTransform)


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
    results = list(Pipeline().run(simple_lf))
    assert len(results) == 1
    assert_frame_equal(results[0].collect(), simple_lf.collect())


def test_run_single_transform(simple_lf: pl.LazyFrame) -> None:
    """Verify that a single transform applies correctly to a LazyFrame."""
    pipe = Pipeline().then(_add_one_to_x)
    result = pipe.run_single(simple_lf).collect()
    expected = simple_lf.with_columns(pl.col("x") + 1).collect()
    assert_frame_equal(result, expected)


def test_run_chained_transforms_apply_in_order(simple_lf: pl.LazyFrame) -> None:
    """Ensure chained transforms apply in the exact order they were added."""
    # add 1 first, then multiply by 2 → (x+1)*2
    pipe = Pipeline().then(_add_one_to_x).then(_multiply_x_by_two)
    result = pipe.run_single(simple_lf).collect()
    expected = simple_lf.with_columns((pl.col("x") + 1) * 2).collect()
    assert_frame_equal(result, expected)


def test_run_fan_out_produces_multiple_outputs(simple_lf: pl.LazyFrame) -> None:
    """Verify that a fan-out transform correctly produces multiple output frames."""
    pipe = Pipeline().then_flat_map(_split_by_id)
    results = list(pipe.run(simple_lf))
    assert len(results) == 2  # two distinct IDs
    # Each should have 4 rows
    for r in results:
        assert r.collect().shape[0] == 4


def test_run_transform_after_fan_out_applied_to_each(simple_lf: pl.LazyFrame) -> None:
    """Ensure transforms added after a fan-out apply independently to all resulting frames."""
    pipe = Pipeline().then_flat_map(_split_by_id).then(_add_one_to_x)
    results = [r.collect() for r in pipe.run(simple_lf)]
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
        pipe.run_single(simple_lf)


def test_run_single_succeeds_for_single_output(simple_lf: pl.LazyFrame) -> None:
    """Ensure run_single succeeds seamlessly when exactly one frame is produced."""
    pipe = Pipeline().then(_add_one_to_x)
    result = pipe.run_single(simple_lf)
    assert isinstance(result, pl.LazyFrame)


def test_compose_then_run(simple_lf: pl.LazyFrame) -> None:
    """Verify that a composed pipeline runs all internal transforms correctly."""
    p1 = Pipeline().then(_add_one_to_x)
    p2 = Pipeline().then(_multiply_x_by_two)
    combined = p1.compose(p2)
    result = combined.run_single(simple_lf).collect()
    expected = simple_lf.with_columns((pl.col("x") + 1) * 2).collect()
    assert_frame_equal(result, expected)


def test_lambda_transform(simple_lf: pl.LazyFrame) -> None:
    """Ensure lambda functions correctly mutate the data within the pipeline."""
    pipe = Pipeline().then(lambda df: df.with_columns(pl.col("x") + 100))
    result = pipe.run_single(simple_lf).collect()
    assert result["x"].min() >= 100  # pyright: ignore[reportOperatorIssue]


# ═══════════════════════════════════════════════════════════════════════════
# Built-in transforms
# ═══════════════════════════════════════════════════════════════════════════


def test_lambda_with_columns(simple_lf: pl.LazyFrame) -> None:
    """Verify a lambda transform using with_columns adds the expected columns."""
    pipe = Pipeline().then(lambda df: df.with_columns(pl.lit(42).alias("answer")))
    result = pipe.run_single(simple_lf).collect()
    assert "answer" in result.columns
    assert result["answer"].unique().to_list() == [42]


def test_lambda_select(simple_lf: pl.LazyFrame) -> None:
    """Verify a lambda transform using select keeps only the specified columns."""
    pipe = Pipeline().then(lambda df: df.select("frame", "id"))
    result = pipe.run_single(simple_lf).collect()
    assert result.columns == ["frame", "id"]


def test_lambda_drop(simple_lf: pl.LazyFrame) -> None:
    """Verify a lambda transform using drop successfully removes specified columns."""
    pipe = Pipeline().then(lambda df: df.drop("y"))
    result = pipe.run_single(simple_lf).collect()
    assert "y" not in result.columns
    assert "x" in result.columns


def test_lambda_rename(simple_lf: pl.LazyFrame) -> None:
    """Verify a lambda transform using rename alters the column names correctly."""
    pipe = Pipeline().then(lambda df: df.rename({"x": "pos_x", "y": "pos_y"}))
    result = pipe.run_single(simple_lf).collect()
    assert "pos_x" in result.columns
    assert "pos_y" in result.columns
    assert "x" not in result.columns


def test_lambda_filter(simple_lf: pl.LazyFrame) -> None:
    """Verify a lambda transform using filter retains only the correct rows."""
    pipe = Pipeline().then(lambda df: df.filter(pl.col("id") == 1))
    result = pipe.run_single(simple_lf).collect()
    assert result["id"].unique().to_list() == [1]
    assert result.shape[0] == 4


def test_lambda_sort() -> None:
    """Verify a lambda transform using sort properly orders the rows."""
    lf = pl.DataFrame({"a": [3, 1, 2]}).lazy()
    pipe = Pipeline().then(lambda df: df.sort("a"))
    assert pipe.run_single(lf).collect()["a"].to_list() == [1, 2, 3]


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
    config = LoaderConfig(3, 3, 0.1)
    fn = transform.filter(config)
    result = fn(trajectory_lf).collect()
    assert_frame_equal(result, trajectory_lf.collect())


def test_transform_filter_removes_category(trajectory_lf: pl.LazyFrame) -> None:
    """Verify the filter transform removes agents that match the specified filtering category."""
    # Filter out agent_category == 1 -> should remove all

    config = LoaderConfig(3, 3, 0.1).with_filtering(filter_agent_category=[AgentCategory.CAR])
    fn = transform.filter(config)
    result = fn(trajectory_lf).collect()
    # All agents have category 1 (CAR), and min_agents=2 by default,
    # so after filtering CAR agents out, no valid agents remain,
    # thus the entire frame should be filtered
    assert result.shape[0] == 0


def test_transform_resample_identity(simple_lf: pl.LazyFrame) -> None:
    """Ensure resampling with identical rates accurately preserves the row count."""
    config = LoaderConfig(2, 2, 1.0)
    fn = transform.resample(config, group_by="id")
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
    config = LoaderConfig(5, 5, 0.1).with_resampling(1, 2)
    fn = transform.resample(config)
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
    config = LoaderConfig(3, 2, 0.1).with_window(step_size=2, window_size=5)
    fan = transform.window(config)
    results = list(fan(lf))
    assert len(results) >= 2


def test_transform_window_offsets_frame() -> None:
    """Verify the window transform completely resets the starting frame offset to zero."""
    lf = pl.DataFrame({
        "frame": list(range(10)),
        "x": [float(i) for i in range(10)],
    }).lazy()
    config = LoaderConfig(3, 2, 0.1).with_window(step_size=3, window_size=5)
    fan = transform.window(config)
    for result_lf in fan(lf):
        result = result_lf.collect()
        # Frame should be zero-offset
        assert result["frame"].min() == 0


def test_transform_window_no_offset() -> None:
    """Verify the window transform retains the original frame offset when configured not to offset."""
    lf = pl.DataFrame({
        "frame": list(range(6)),
        "x": [float(i) for i in range(6)],
    }).lazy()
    config = LoaderConfig(3, 2, 0.1).with_window(step_size=3, window_size=5)
    fan = transform.window(config, offset_sliding_col=False)
    results = [r.collect() for r in fan(lf)]
    # At least first window should start at frame 0
    assert results[0]["frame"].min() == 0


def test_transform_window_requires_config() -> None:
    """Ensure the window transform raises an error if the window parameters are missing."""
    config = LoaderConfig(3, 2, 0.1)  # No window_params
    with pytest.raises(ValueError, match="window_params"):
        transform.window(config)


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
    result = pipe.run_single(lf).collect()
    assert result.shape[0] == 2
    assert result["yaw"][0] == pytest.approx(0.0)


def test_pipeline_integration_window_then_transform() -> None:
    """Apply a windowing fan-out followed immediately by a custom transform on each window."""
    lf = pl.DataFrame({
        "frame": list(range(10)),
        "x": [float(i) for i in range(10)],
        "y": [0.0] * 10,
    }).lazy()
    config = LoaderConfig(3, 2, 0.1).with_window(step_size=3, window_size=5)
    pipe = (
        Pipeline()
        .then_flat_map(transform.window(config))
        .then(lambda df: df.with_columns(pl.lit(99).alias("marker")))
    )
    results = [r.collect() for r in pipe.run(lf)]
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
    results = [r.collect() for r in pipe.run(lf)]
    assert len(results) == 2
    for r in results:
        assert "pos_x" in r.columns
        assert r["pos_x"].min() >= 100.0  # pyright: ignore[reportOperatorIssue]


# ═══════════════════════════════════════════════════════════════════════════
# pipeline_from_config
# ═══════════════════════════════════════════════════════════════════════════


def test_pipeline_from_config_basic_no_window_no_resample() -> None:
    """Create and run a pipeline directly from configuration without windowing or resampling."""
    config = LoaderConfig(3, 3, 0.1).with_filtering(min_agents=1)
    pipe = pipeline_from_config(config)
    assert len(pipe) > 0

    lf = pl.DataFrame({
        "frame": [0, 1, 2, 3, 4, 5],
        "id": [1, 1, 1, 1, 1, 1],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "agent_category": [1, 1, 1, 1, 1, 1],
    }).lazy()

    results = list(pipe.run(lf))
    assert len(results) == 1
    result = results[0].collect()
    assert result.shape[0] == 6


def test_pipeline_from_config_with_window_produces_fan_out() -> None:
    """Create and run a pipeline from configuration to successfully produce multiple windows."""
    config = (
        LoaderConfig(3, 3, 0.1).with_window(step_size=2, window_size=6).with_filtering(min_agents=1)
    )
    pipe = pipeline_from_config(config)

    lf = pl.DataFrame({
        "frame": list(range(12)),
        "id": [1] * 12,
        "x": [float(i) for i in range(12)],
        "y": [0.0] * 12,
        "agent_category": [1] * 12,
    }).lazy()

    results = list(pipe.run(lf))
    assert len(results) >= 2


def test_pipeline_from_config_with_yaw_from_vel() -> None:
    """Create a pipeline from configuration that accurately calculates yaw, velocity, and acceleration."""
    config = LoaderConfig(3, 3, 0.1).with_filtering(min_agents=1)
    pipe = pipeline_from_config(
        config,
        add_derivative=True,
        add_second_derivative=True,
        derivative_rename={1: ["vx", "vy"], 2: ["ax", "ay"]},
        add_yaw_from_vel=True,
    )

    lf = pl.DataFrame({
        "frame": [0, 1, 2, 3, 4, 5],
        "id": [1, 1, 1, 1, 1, 1],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "agent_category": [1, 1, 1, 1, 1, 1],
    }).lazy()

    result = next(iter(pipe.run(lf))).collect()
    assert "yaw" in result.columns
    assert "vx" in result.columns
    assert "ax" in result.columns


def test_pipeline_from_config_no_category_column() -> None:
    """Verify pipeline creation functions correctly even if no category column is specified."""
    config = LoaderConfig(2, 2, 0.1).with_filtering(min_agents=1)
    pipe = pipeline_from_config(config, category_column=None)

    lf = pl.DataFrame({
        "frame": [0, 1, 2, 3],
        "id": [1, 1, 1, 1],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 0.0, 0.0, 0.0],
    }).lazy()

    results = list(pipe.run(lf))
    assert len(results) == 1


# ═══════════════════════════════════════════════════════════════════════════
# BaseSceneLoader integration (pipeline vs normalize fallback)
# ═══════════════════════════════════════════════════════════════════════════


def test_loader_legacy_normalize_fallback() -> None:
    """Ensure the legacy normalize fallback method triggers when the pipeline is empty."""

    class LegacyLoader(BaseSceneLoader[int, pl.LazyFrame]):
        @override
        def sources(self) -> Iterable[Source[int, pl.LazyFrame]]:
            yield Source(
                identifier=0,
                inner=pl.DataFrame({
                    "frame": [0, 1, 2, 3],
                    "id": [1, 1, 1, 1],
                    "x": [0.0, 1.0, 2.0, 3.0],
                    "y": [0.0, 0.0, 0.0, 0.0],
                    "vx": [1.0, 1.0, 1.0, 1.0],
                    "vy": [0.0, 0.0, 0.0, 0.0],
                    "ax": [0.0, 0.0, 0.0, 0.0],
                    "ay": [0.0, 0.0, 0.0, 0.0],
                    "yaw": [0.0, 0.0, 0.0, 0.0],
                    "agent_category": [1, 1, 1, 1],
                }).lazy(),
            )

        @override
        def load_raw(
            self, source: Source[int, pl.LazyFrame]
        ) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
            yield source.inner, mc.NoMap()

        @override
        def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(pl.col("x") + 1000)

        @classmethod
        def default_config(cls) -> LoaderConfig:
            return LoaderConfig(2, 2, 0.1)

    loader = LegacyLoader()
    scenes = list(loader.scenes())
    assert len(scenes) == 1
    # normalize should have been called
    assert scenes[0].inner["x"].min() >= 1000  # pyright: ignore[reportOperatorIssue]


def test_loader_pipeline_overrides_normalize() -> None:
    """Ensure the pipeline execution logic correctly overrides the normalize method if present."""

    class PipelineLoader(BaseSceneLoader[int, pl.LazyFrame]):
        @override
        def sources(self) -> Iterable[Source[int, pl.LazyFrame]]:
            yield Source(
                identifier=0,
                inner=pl.DataFrame({
                    "frame": [0, 1, 2, 3],
                    "id": [1, 1, 1, 1],
                    "x": [0.0, 1.0, 2.0, 3.0],
                    "y": [0.0, 0.0, 0.0, 0.0],
                    "vx": [1.0, 1.0, 1.0, 1.0],
                    "vy": [0.0, 0.0, 0.0, 0.0],
                    "ax": [0.0, 0.0, 0.0, 0.0],
                    "ay": [0.0, 0.0, 0.0, 0.0],
                    "yaw": [0.0, 0.0, 0.0, 0.0],
                    "agent_category": [1, 1, 1, 1],
                }).lazy(),
            )

        @override
        def load_raw(
            self, source: Source[int, pl.LazyFrame]
        ) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
            yield source.inner, mc.NoMap()

        @override
        def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
            # This should NOT be called
            return df.with_columns(pl.col("x") + 9999)

        @override
        def pipeline(self) -> Pipeline:
            return Pipeline().then(lambda df: df.with_columns(pl.col("x") + 1))

        @classmethod
        def default_config(cls) -> LoaderConfig:
            return LoaderConfig(2, 2, 0.1)

    loader = PipelineLoader()
    scenes = list(loader.scenes())
    assert len(scenes) == 1
    # pipeline adds 1, normalize would add 9999
    assert scenes[0].inner["x"].max() == pytest.approx(4.0)


def test_loader_pipeline_fan_out_creates_multiple_scenes() -> None:
    """Verify that a pipeline fan-out accurately creates multiple scenes from a single source."""

    class FanOutLoader(BaseSceneLoader[int, pl.LazyFrame]):
        @override
        def sources(self) -> Iterable[Source[int, pl.LazyFrame]]:
            yield Source(
                identifier=0,
                inner=pl.DataFrame({
                    "frame": list(range(8)),
                    "id": [1] * 4 + [2] * 4,
                    "x": [float(i) for i in range(8)],
                    "y": [0.0] * 8,
                    "vx": [1.0] * 8,
                    "vy": [0.0] * 8,
                    "ax": [0.0] * 8,
                    "ay": [0.0] * 8,
                    "yaw": [0.0] * 8,
                    "agent_category": [1] * 8,
                }).lazy(),
            )

        @override
        def load_raw(
            self, source: Source[int, pl.LazyFrame]
        ) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
            yield source.inner, mc.NoMap()

        @override
        def pipeline(self) -> Pipeline:
            return Pipeline().then_flat_map(transform.group_by_yield("id", drop_group_cols=False))

        @classmethod
        def default_config(cls) -> LoaderConfig:
            return LoaderConfig(2, 2, 0.1)

    loader = FanOutLoader(enforce_schema=False)
    scenes = list(loader.scenes())
    assert len(scenes) == 2
    # Both scenes share the same source identifier
    assert all(s.identifier == 0 for s in scenes)
    # Scene numbers should be sequential
    scene_numbers = {s.scene_number for s in scenes}
    assert scene_numbers == {0, 1}
