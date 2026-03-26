"""Tests for cumulative_blocks and shuffled_blocks."""

from __future__ import annotations

import polars as pl
import pytest

from dronalize.pipeline.functional.block import cumulative_blocks, shuffled_blocks


def test_cumulative_blocks_split_evenly_with_gap_and_offset() -> None:
    """Split rows into contiguous blocks and offset time."""
    df = pl.DataFrame({
        "frame": list(range(8)),
        "value": [10, 11, 12, 13, 20, 21, 22, 23],
    })

    result = cumulative_blocks(
        df,
        weights=[0.5, 0.5],
        time_column="frame",
        gap=2,
    )

    expected = pl.DataFrame({
        "frame": [0, 1, 2, 0, 1, 2],
        "value": [10, 11, 12, 21, 22, 23],
        "block": [0, 0, 0, 1, 1, 1],
    })

    assert result.equals(expected)


def test_cumulative_blocks_keep_gap_rows_when_requested() -> None:
    """Keep gap rows and preserve null block assignments."""
    df = pl.DataFrame({
        "frame": list(range(8)),
        "value": [10, 11, 12, 13, 20, 21, 22, 23],
    })

    result = cumulative_blocks(
        df,
        weights=[0.5, 0.5],
        time_column="frame",
        gap=2,
        remove_gap=False,
        offset_time_column=False,
    )

    expected = pl.DataFrame({
        "frame": list(range(8)),
        "value": [10, 11, 12, 13, 20, 21, 22, 23],
        "block": [0, 0, 0, None, None, 1, 1, 1],
    })

    assert result.equals(expected)


def test_cumulative_blocks_apply_per_group() -> None:
    """Assign blocks independently within each group."""
    df = pl.DataFrame({
        "scene": ["A"] * 6 + ["B"] * 6,
        "frame": [0, 1, 2, 3, 4, 5] * 2,
        "value": list(range(12)),
    })

    result = cumulative_blocks(
        df,
        weights=[0.5, 0.5],
        time_column="frame",
        group_by="scene",
        gap=0,
    ).sort(["scene", "frame", "value"])

    expected = pl.DataFrame({
        "scene": ["A"] * 6 + ["B"] * 6,
        "frame": [0, 1, 2, 0, 1, 2] * 2,
        "value": list(range(12)),
        "block": [0, 0, 0, 1, 1, 1] * 2,
    }).sort(["scene", "frame", "value"])

    assert result.equals(expected)


def test_cumulative_blocks_raise_when_gap_exceeds_available_frames() -> None:
    """Reject impossible block layouts."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "value": [10, 11, 12],
    })

    with pytest.raises(ValueError, match="leaves no usable frames"):
        _ = cumulative_blocks(
            df,
            weights=[1.0, 1.0, 1.0],
            time_column="frame",
            gap=2,
        )


def test_shuffled_blocks_be_deterministic_for_fixed_seed() -> None:
    """Produce stable assignments for a fixed seed."""
    df = pl.DataFrame({
        "frame": list(range(12)),
        "value": list(range(12)),
    })

    result_1 = shuffled_blocks(
        df,
        weights=[0.6, 0.4],
        n_segments=6,
        time_column="frame",
        seed=123,
        assignment_column="unit",
    ).sort(["frame", "value"])

    result_2 = shuffled_blocks(
        df,
        weights=[0.6, 0.4],
        n_segments=6,
        time_column="frame",
        seed=123,
        assignment_column="unit",
    ).sort(["frame", "value"])

    assert result_1.equals(result_2)


def test_shuffled_blocks_keep_assignment_column_and_add_segment_column() -> None:
    """Return both contiguous segments and weighted assignments."""
    df = pl.DataFrame({
        "frame": list(range(12)),
        "value": list(range(12)),
    })

    result = shuffled_blocks(
        df,
        weights=[0.5, 0.3, 0.2],
        n_segments=4,
        time_column="frame",
        seed=7,
        assignment_column="unit",
        segment_column="_u",
    )

    assert "unit" in result.columns

    unique_units = result.select(pl.col("unit").drop_nulls().n_unique()).item()
    unique_assignments = set(result["unit"].to_list())

    assert unique_units == 3
    print(unique_assignments)
    assert unique_assignments.issubset({0, 1, 2})


def test_shuffled_blocks_assign_segment_consistently_within_each_group() -> None:
    """Assign the same label to all rows of a segment within a group."""
    df = pl.DataFrame({
        "scene": ["A"] * 8 + ["B"] * 8,
        "frame": [0, 1, 2, 3, 4, 5, 6, 7] * 2,
        "value": list(range(16)),
    })

    result = shuffled_blocks(
        df,
        weights=[0.5, 0.5],
        n_segments=4,
        time_column="frame",
        group_by="scene",
        seed=42,
        assignment_column="unit",
    )

    per_unit = (
        result
        .group_by(["scene", "unit"])
        .agg(pl.col("unit").n_unique().alias("n_assignments"))
        .sort(["scene", "unit"])
    )

    assert per_unit["n_assignments"].to_list() == [1] * per_unit.height
