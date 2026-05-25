"""Built-in transform /fan-out factory functions for pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from dronalize.core import functional as f
from dronalize.core.functional import ResampleSpec
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.screening.screen import screen_scene as _screen_scene

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dronalize.core.functional.window import WindowPolicy
    from dronalize.processing.pipeline.pipeline import FlatMapTransform, Transform
    from dronalize.processing.screening import ScreeningRuleSet


def screen_scene(
    scene_screening: ScreeningRuleSet | None = None,
    columns: TrajectoryColumns | None = None,
    *,
    group_by: str | Sequence[str] | None = None,
    mark_passed_agents: bool = False,
    retain_scene_passes: bool = False,
) -> Transform:
    """Create a screening transform.

    Wraps [`dronalize.processing.screening.screen_scene`][].

    Parameters
    ----------
    scene_screening : ScreeningRuleSet, optional
        Screening specification containing cleanup and check rules.
    columns : TrajectoryColumns, optional
        Column mapping for the input frame.
    group_by : str or Sequence[str], optional
        Column(s) that define independent scenes inside the frame.
    mark_passed_agents : bool, optional
        Whether to retain an internal per-agent passed marker for runtime transport.

    """

    def _screen(df: pl.LazyFrame) -> pl.LazyFrame:
        return _screen_scene(
            df,
            scene_screening=scene_screening,
            columns=columns or TrajectoryColumns(),
            scene_group_by=group_by,
            mark_passed_agents=mark_passed_agents,
            retain_scene_passes=retain_scene_passes,
        )

    _screen.__name__ = "screen"
    _screen.__qualname__ = "transforms.screen"
    return _screen


def resample(
    spec: ResampleSpec | None = None,
    *,
    frame_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
) -> Transform:
    """Create a resampling transform.

    Parameters
    ----------
    spec : ResampleSpec, optional
        Resampling specification.
    frame_column : str, optional
        Column name for the frame index.
    group_by : str or Sequence[str], optional
        Columns to partition tracks by.

    """
    resample_spec = spec or ResampleSpec()

    def _resample(df: pl.LazyFrame) -> pl.LazyFrame:
        return f.resample(df, resample_spec, frame_column=frame_column, group_by=group_by)

    _resample.__name__ = "resample"
    _resample.__qualname__ = "transforms.resample"
    return _resample


def derivative(
    *columns: str,
    sample_time: float = 1.0,
    order: int = 1,
    include_intermediate: bool = False,
    group_by: str | Sequence[str] | None = None,
    derivative_rename: dict[int, list[str]] | None = None,
) -> Transform:
    """Create a derivative transform.

    Wraps `~dronalize.common.trajectory.derivative.derivative`.

    Parameters
    ----------
    *columns : str
        Column names to differentiate.
    sample_time : float, optional
        Time step.
    order : int, optional
        Derivative order.
    include_intermediate : bool, optional
        Keep all intermediate derivative orders.
    group_by : str or Sequence[str], optional
        Partition columns.
    derivative_rename : dict[int, list[str]], optional
        Custom derivative column names.

    """

    def _derivative(df: pl.LazyFrame) -> pl.LazyFrame:
        return f.derivative(
            df,
            *columns,
            sample_time=sample_time,
            order=order,
            include_intermediate=include_intermediate,
            group_by=group_by,
            derivative_rename=derivative_rename,
        )

    _derivative.__name__ = "derivative"
    _derivative.__qualname__ = "transforms.derivative"
    return _derivative


def yaw_from_velocity(
    vx_col: str = "vx", vy_col: str = "vy", yaw_col: str = "yaw", *, only_null: bool = False
) -> Transform:
    """Create a yaw-from-velocity transform.

    Parameters
    ----------
    vx_col : str, optional
        X-velocity column name.
    vy_col : str, optional
        Y-velocity column name.
    yaw_col : str, optional
        Output yaw column name.
    only_null : bool, optional
        Whether to only compute yaw for rows where `yaw_col` is null. Note that
        this requires that `yaw_col` already exists in the input frame.

    """
    if only_null:

        def _yaw_from_velocity(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(
                pl
                .when(pl.col(yaw_col).is_null())
                .then(f.yaw_from_velocity_expr(vx_col, vy_col, yaw_col))
                .otherwise(pl.col(yaw_col))
                .alias(yaw_col)
            )

    else:

        def _yaw_from_velocity(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(f.yaw_from_velocity_expr(vx_col, vy_col, yaw_col))

    _yaw_from_velocity.__name__ = "yaw_from_velocity"
    _yaw_from_velocity.__qualname__ = "transforms.yaw_from_velocity"
    return _yaw_from_velocity


def yaw_from_position(
    x_col: str = "x", y_col: str = "y", yaw_col: str = "yaw", *, only_null: bool = False
) -> Transform:
    """Create a yaw-from-position transform.

    Wraps `~dronalize.common.trajectory.basic.yaw_from_position`.

    Parameters
    ----------
    x_col : str, optional
        X-position column name.
    y_col : str, optional
        Y-position column name.
    yaw_col : str, optional
        Output yaw column name.
    only_null : bool, optional
        Whether to only compute yaw for rows where `yaw_col` is null. Note that
        this requires that `yaw_col` already exists in the input frame.

    """
    if only_null:

        def _yaw_from_position(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(
                pl
                .when(pl.col(yaw_col).is_null())
                .then(f.yaw_from_position_expr(x_col, y_col, yaw_col))
                .otherwise(pl.col(yaw_col))
                .alias(yaw_col)
            )

    else:

        def _yaw_from_position(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(f.yaw_from_position_expr(x_col, y_col, yaw_col))

    _yaw_from_position.__name__ = "yaw_from_position"
    _yaw_from_position.__qualname__ = "transforms.yaw_from_position"
    return _yaw_from_position


def window(
    window_size: int,
    step_size: int,
    *,
    sliding_col: str = "frame",
    group_by: str | Sequence[str] | None = None,
    offset_sliding_col: bool = True,
    policy: WindowPolicy = "strict",
) -> Transform:
    """Create a sliding-window fan-out transform.

    This is a 1:N step: a single LazyFrame is split into many overlapping
    windows, each yielded as a separate LazyFrame.

    Parameters
    ----------
    window_size : int
        Number of rows in each window.
    step_size : int
        Number of rows to move the window at each step.
    sliding_col : str, optional
        Column to slide over.
    offset_sliding_col : bool, optional
        Whether to zero-offset the sliding column in each window.
    policy : WindowPolicy, optional
        Windowing policy to apply when the last window does not have enough rows.

    """

    def _window_single(df: pl.LazyFrame) -> pl.LazyFrame:
        return f.sliding_window(
            df,
            window_size=window_size,
            step_size=step_size,
            sliding_col=sliding_col,
            group_by=group_by,
            offset_sliding_col=offset_sliding_col,
            policy=policy,
        )

    func_to_return = _window_single

    # Reassign names dynamically to the chosen function
    func_to_return.__name__ = "window"
    func_to_return.__qualname__ = "transforms.window"

    return func_to_return


def block_partition_cumulative(
    weights: Sequence[float],
    *,
    time_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
    gap: int = 0,
    remove_gap: bool = True,
    offset_time_column: bool = True,
    partition_column: str = "block",
) -> Transform:
    """Partition the time dimension into contiguous temporal partitions.

    This assigns a partition id to each row based on cumulative time and the
    provided weights. Optionally, a gap can be inserted between partitions, and
    the original time column can be offset to start from zero inside each
    partition.

    Parameters
    ----------
    weights : Sequence[float]
        Relative weights for each partition. The number of partitions is determined by
        the length of this sequence.
    time_column : str, optional
        Name of the time column used for temporal partitioning.
    group_by : str or Sequence[str] or None, optional
        Column name(s) used to partition each group independently.
    gap : int, optional
        Number of excluded timesteps inserted between adjacent partitions.
    remove_gap : bool, optional
        Whether to remove rows that fall into the excluded gap. Default is
        True.
    offset_time_column : bool, optional
        Whether to offset the time column to start from zero inside each
        partition.
        Default is True.
    partition_column : str, optional
        Name of the output partition column. Default is `"block"`.

    """

    def _block_partition_cumulative(df: pl.LazyFrame) -> pl.LazyFrame:
        return f.cumulative_blocks(
            df,
            weights=weights,
            time_column=time_column,
            group_by=group_by,
            gap=gap,
            remove_gap=remove_gap,
            offset_time_column=offset_time_column,
            partition_column=partition_column,
        )

    _block_partition_cumulative.__name__ = "block_partition_cumulative"
    _block_partition_cumulative.__qualname__ = "transforms.block_partition_cumulative"
    return _block_partition_cumulative


def block_partition_shuffle(
    weights: Sequence[float],
    segments: int,
    *,
    time_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
    gap: int = 0,
    seed: int | None = None,
    offset_time_column: bool = True,
    assignment_column: str = "block",
    segment_column: str = "unit",
) -> Transform:
    """Partition the time dimension into shuffled temporal segments.

    Parameters
    ----------
    weights : Sequence[float]
        Relative weights for each output assignment. The number of assignments is determined by
        the length of this sequence.
    segments : int
        Number of contiguous temporal segments to create before routing them to
        shuffled assignments.
    time_column : str, optional
        Name of the time column used for temporal partitioning.
    group_by : str or Sequence[str] or None, optional
        Column name(s) used to partition each group independently.
    gap : int, optional
        Number of excluded timesteps inserted between adjacent segments.
    seed : int or None, optional
        Random seed used for the deterministic weighted assignment.
    offset_time_column : bool, optional
        Whether to offset the time column to start from zero inside each
        segment.
        Default is True.
    assignment_column : str, optional
        Name of the output assignment column. Default is `"block"`.
    segment_column : str, optional
        Name of the intermediate contiguous segment column. Default is `"unit"`.

    """

    def _block_partition_shuffle(df: pl.LazyFrame) -> pl.LazyFrame:
        return f.shuffled_blocks(
            df,
            weights=weights,
            n_segments=segments,
            time_column=time_column,
            group_by=group_by,
            gap=gap,
            seed=seed,
            offset_time_column=offset_time_column,
            assignment_column=assignment_column,
            segment_column=segment_column,
        )

    _block_partition_shuffle.__name__ = "block_partition_shuffle"
    _block_partition_shuffle.__qualname__ = "transforms.block_partition_shuffle"
    return _block_partition_shuffle


def valid_lane_change(
    persist: int,
    margin_before: int = 0,
    margin_after: int = 0,
    *,
    frame_column: str = "frame",
    agent_id_column: str = "id",
    lane_id_column: str = "lane_id",
    group_by: str | Sequence[str] | None = None,
    valid_column: str = "valid_lane_change",
) -> Transform:
    """Add a boolean column indicating whether change occurs at each frame.

    This is a wrapper around `valid_lane_change_expr` that handles the necessary
    sorting to ensure correct results. The added column will be named according
    to the `valid_column` parameter (default is "valid_lane_change"). See
    `valid_lane_change_expr` for more details.

    Parameters
    ----------
    persist : int
        Number of frames the lane change should persist in order to be considered
        valid. This is useful if the lane assignment is noisy.
    margin_before : int, optional
        Number of frames that should exist before the lane change in order
        to be considered valid.
    margin_after : int, optional
        Number of frames that should exist after the lane change in order
        to be considered valid.
    frame_column : str, optional
        Name of the frame column. Default is "frame".
    agent_id_column : str, optional
        Name of the agent id column. Default is "id".
    lane_id_column : str, optional
        Name of the lane id column. Default is "lane_id".
    group_by : str or Sequence[str] or None, optional
        Column name(s) used to partition each group independently. Default is no
        partitioning.
    valid_column : str, optional
        Name of the output boolean column indicating valid lane changes.

    """

    def _valid_lane_change(df: pl.LazyFrame) -> pl.LazyFrame:
        return f.valid_lane_change(
            df,
            persist=persist,
            margin_before=margin_before,
            margin_after=margin_after,
            frame_column=frame_column,
            agent_id_column=agent_id_column,
            lane_id_column=lane_id_column,
            group_by=group_by,
            valid_column=valid_column,
        )

    _valid_lane_change.__name__ = "valid_lane_change"
    _valid_lane_change.__qualname__ = "transforms.valid_lane_change"
    return _valid_lane_change


def group_by_yield(*by: str, drop_group_cols: bool = True) -> FlatMapTransform:
    """Create a fan-out that partitions by column(s) and yields each group.

    This collects the LazyFrame, groups it, and yields each group as
    a new LazyFrame.  Useful for datasets that pack multiple scenes
    into a single file (e.g., batched Parquet loading in Argoverse).

    Parameters
    ----------
    *by : str
        Column names to group by.
    drop_group_cols : bool, optional
        Whether to drop the grouping columns from each yielded frame.

    """
    by_tuple = tuple(by)

    def _group_by_yield(df: pl.LazyFrame) -> Iterable[pl.LazyFrame]:
        collected = df.collect()

        parts = collected.partition_by(
            *by_tuple, maintain_order=True, as_dict=False, include_key=not drop_group_cols
        )

        for part in parts:
            yield part.lazy()

    _group_by_yield.__name__ = "group_by_yield"
    _group_by_yield.__qualname__ = "transforms.group_by_yield"
    return _group_by_yield


def select(*expr: pl.Expr, **named_expr: pl.Expr) -> Transform:
    """Create a select transform from given expressions.

    Parameters
    ----------
    *expr : pl.Expr
        Positional expressions to pass to select.
    **named_expr : pl.Expr
        Keyword expressions to pass to select.

    """

    def _select(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.select(*expr, **named_expr)

    _select.__name__ = "select"
    _select.__qualname__ = "transforms.select"
    return _select


def with_columns(*expr: pl.Expr, **named_expr: pl.Expr) -> Transform:
    """Create a with_columns transform from given expressions.

    Parameters
    ----------
    *expr : pl.Expr
        Positional expressions to pass to with_columns.
    **named_expr : pl.Expr
        Keyword expressions to pass to with_columns.

    """

    def _with_columns(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns(*expr, **named_expr)

    _with_columns.__name__ = "with_columns"
    _with_columns.__qualname__ = "transforms.with_columns"
    return _with_columns
