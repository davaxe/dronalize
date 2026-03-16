"""Built-in transform /fan-out factory functions for pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
from typing_extensions import overload

from dronalize.pipeline.functional.basic import yaw_from_pos_expr as _yaw_from_pos_expr
from dronalize.pipeline.functional.basic import yaw_from_vel_expr as _yaw_from_vel_expr
from dronalize.pipeline.functional.derivative import derivative as _derivative_impl
from dronalize.pipeline.functional.filter import filter_scene_expr
from dronalize.pipeline.functional.rebalance import rebalance_highway_agents
from dronalize.pipeline.functional.resample import Resampling
from dronalize.pipeline.functional.resample import resample as resample_impl
from dronalize.pipeline.functional.window import sliding_window

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dronalize.config.filtering import FilteringConfig
    from dronalize.pipeline.pipeline import FlatMapTransform, Transform

__all__ = [
    "derivative",
    "filter_scene",
    "group_by_yield",
    "rebalance",
    "resample",
    "select",
    "window",
    "with_columns",
    "yaw_from_pos",
    "yaw_from_vel",
]


def filter_scene(
    config: FilteringConfig | None = None,
    *,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = "agent_category",
) -> Transform:
    """Create a filtering transform.

    Wraps `~dronalize.common.trajectory.filter.filter_scene_expr`.

    Parameters
    ----------
    config : FilteringConfig, optional
        Configuration containing the filtering criteria.
    group_by : str or Sequence[str], optional
        Column(s) that define independent scenes inside the frame.
    agent_id : str, optional
        Column name for agent IDs.
    frame_column : str, optional
        Column name for frame indices.
    category_column : str, optional
        Column name for agent categories.

    Returns
    -------
    Transform

    """
    expr = filter_scene_expr(
        config=config,
        group_by=group_by,
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
    )

    def _filter(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(expr)

    _filter.__name__ = "filter"
    _filter.__qualname__ = "transforms.filter"
    return _filter


def require_min(group_by: str | Sequence[str], minimum: int = 2) -> Transform:
    """Require a minimum number of rows per group.

    Parameters
    ----------
    group_by : str or Sequence[str]
        Column(s) to group by.
    minimum : int, optional
        Minimum number of rows required per group. Default is 1.

    Returns
    -------
    Transform
    """

    def _require_min(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(pl.len().over(group_by) >= minimum)

    _require_min.__name__ = "require_min"
    _require_min.__qualname__ = "transforms.require_min"
    return _require_min


def resample(
    resampling: Resampling | None = None,
    *,
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    velocity_columns: Sequence[str] = (),
    acceleration_columns: Sequence[str] = (),
    group_by: str | Sequence[str] | None = None,
) -> Transform:
    """Create a resampling transform.

    Wraps `~dronalize.common.trajectory.resample.resample_tracks`.
    If the resampling object specifies no resampling (ratio 1:1), the transform
    is still applied so that the same interpolation and zero-order-hold rules
    are used consistently across pipelines.

    Parameters
    ----------
    resampling : Resampling, optional
        Resampling parameters.
    frame_column : str, optional
        Column name for the frame index.
    pos_columns : Sequence[str], optional
        Position column names.
    velocity_columns : Sequence[str], optional
        Velocity columns preserved using zero-order hold.
    acceleration_columns : Sequence[str], optional
        Acceleration columns preserved using zero-order hold.
    group_by : str or Sequence[str], optional
        Columns to partition tracks by.

    Returns
    -------
    Transform

    """
    resampling_obj = resampling or Resampling(up=1, down=1)

    def _resample(df: pl.LazyFrame) -> pl.LazyFrame:
        return resample_impl(
            df,
            resampling_obj,
            frame_column=frame_column,
            pos_columns=pos_columns,
            velocity_columns=velocity_columns,
            acceleration_columns=acceleration_columns,
            group_by=group_by,
        )

    _resample.__name__ = "resample"
    _resample.__qualname__ = "transforms.resample"
    return _resample


def derivative(
    *columns: str,
    dt: float = 1.0,
    n: int = 1,
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
    dt : float, optional
        Time step.
    n : int, optional
        Derivative order.
    include_intermediate : bool, optional
        Keep all intermediate derivative orders.
    group_by : str or Sequence[str], optional
        Partition columns.
    derivative_rename : dict[int, list[str]], optional
        Custom derivative column names.

    Returns
    -------
    Transform

    """

    def _derivative(df: pl.LazyFrame) -> pl.LazyFrame:
        return _derivative_impl(
            df,
            *columns,
            dt=dt,
            n=n,
            include_intermediate=include_intermediate,
            group_by=group_by,
            derivative_rename=derivative_rename,
        )

    _derivative.__name__ = "derivative"
    _derivative.__qualname__ = "transforms.derivative"
    return _derivative


def yaw_from_vel(
    vx_col: str = "vx",
    vy_col: str = "vy",
    yaw_col: str = "yaw",
    *,
    only_null: bool = False,
) -> Transform:
    """Create a yaw-from-velocity transform.

    Wraps `~dronalize.common.trajectory.basic.yaw_from_vel`.

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

    Returns
    -------
    Transform

    """
    if only_null:

        def _yaw_from_vel(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(
                pl
                .when(pl.col(yaw_col).is_null())
                .then(_yaw_from_vel_expr(vx_col, vy_col, yaw_col))
                .otherwise(pl.col(yaw_col))
                .alias(yaw_col),
            )

    else:

        def _yaw_from_vel(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(_yaw_from_vel_expr(vx_col, vy_col, yaw_col))

    _yaw_from_vel.__name__ = "yaw_from_vel"
    _yaw_from_vel.__qualname__ = "transforms.yaw_from_vel"
    return _yaw_from_vel


def yaw_from_pos(
    x_col: str = "x",
    y_col: str = "y",
    yaw_col: str = "yaw",
    *,
    only_null: bool = False,
) -> Transform:
    """Create a yaw-from-position transform.

    Wraps `~dronalize.common.trajectory.basic.yaw_from_pos`.

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

    Returns
    -------
    Transform

    """
    if only_null:

        def _yaw_from_pos(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(
                pl
                .when(pl.col(yaw_col).is_null())
                .then(_yaw_from_pos_expr(x_col, y_col, yaw_col))
                .otherwise(pl.col(yaw_col))
                .alias(yaw_col),
            )

    else:

        def _yaw_from_pos(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(_yaw_from_pos_expr(x_col, y_col, yaw_col))

    _yaw_from_pos.__name__ = "yaw_from_pos"
    _yaw_from_pos.__qualname__ = "transforms.yaw_from_pos"
    return _yaw_from_pos


@overload
def window(
    window_size: int,
    step_size: int,
    *,
    sliding_col: str = "frame",
    offset_sliding_col: bool = True,
    return_iterable: Literal[False] = False,
) -> Transform: ...


@overload
def window(
    window_size: int,
    step_size: int,
    *,
    sliding_col: str = "frame",
    offset_sliding_col: bool = True,
    return_iterable: Literal[True],
) -> FlatMapTransform: ...


def window(
    window_size: int,
    step_size: int,
    *,
    sliding_col: str = "frame",
    offset_sliding_col: bool = True,
    return_iterable: bool = False,
) -> FlatMapTransform | Transform:
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

    Returns
    -------
    FlatMapTransform or Transform
        Depending on `return_iterable`, either a transform that returns a single
        frame with a window index column, or a flat-map transform that yields
        each window as a separate frame.

    """
    if return_iterable:

        def _window_iter(df: pl.LazyFrame) -> Iterable[pl.LazyFrame]:
            for window_df in sliding_window(
                df,
                window_size=window_size,
                step_size=step_size,
                sliding_col=sliding_col,
                return_iterable=True,
            ):
                if offset_sliding_col:
                    window_df = window_df.with_columns(  # noqa: PLW2901
                        pl.col(sliding_col) - pl.col(sliding_col).min(),
                    )
                yield window_df.lazy()

        func_to_return = _window_iter

    else:

        def _window_single(df: pl.LazyFrame) -> pl.LazyFrame:
            out = sliding_window(
                df,
                window_size=window_size,
                step_size=step_size,
                sliding_col=sliding_col,
                return_iterable=False,
            )
            if offset_sliding_col:
                out = out.with_columns(pl.col(sliding_col) - pl.col(sliding_col).min())
            return out

        func_to_return = _window_single

    # Reassign names dynamically to the chosen function
    func_to_return.__name__ = "window"
    func_to_return.__qualname__ = "transforms.window"

    return func_to_return


def rebalance(
    ratio: float,
    *,
    req_lane_changes: int = 1,
    agent_id: str = "id",
    lane_changes_col: str = "lane_changes",
    seed: int | None = None,
    drop_lanechange_col: bool = True,
) -> Transform:
    """Create a highway agent rebalancing transform.

    Wraps `~dronalize.common.trajectory.rebalance.rebalance_highway_agents`.

    Parameters
    ----------
    ratio : float
        Target LC/LK ratio.
    req_lane_changes : int, optional
        Minimum lane changes to count as LC agent.
    agent_id : str, optional
        Agent ID column.
    lane_changes_col : str, optional
        Lane-change count column.
    seed : int, optional
        Random seed for reproducibility.
    drop_lanechange_col : bool, optional
        Whether to drop the lane-change column after rebalancing.

    Returns
    -------
    Transform

    """

    def _rebalance(df: pl.LazyFrame) -> pl.LazyFrame:
        result = rebalance_highway_agents(
            df,
            ratio=ratio,
            req_lane_changes=req_lane_changes,
            agent_id=agent_id,
            lane_changes_col=lane_changes_col,
            seed=seed,
        )
        if drop_lanechange_col:
            result = result.drop(lane_changes_col)
        return result

    _rebalance.__name__ = "rebalance"
    _rebalance.__qualname__ = "transforms.rebalance"
    return _rebalance


# -------------------------------------------------------------------
# Group-then-yield (fan-out by column)
# -------------------------------------------------------------------


def group_by_yield(
    *by: str,
    drop_group_cols: bool = True,
) -> FlatMapTransform:
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

    Returns
    -------
    FlatMapTransform

    """
    by_list = list(by)

    def _group_by_yield(df: pl.LazyFrame) -> Iterable[pl.LazyFrame]:
        collected = df.collect()
        for _, group in collected.group_by(by_list):
            out = group.lazy()
            if drop_group_cols:
                out = out.drop(*by_list)
            yield out

    _group_by_yield.__name__ = "group_by_yield"
    _group_by_yield.__qualname__ = "transforms.group_by_yield"
    return _group_by_yield


# --------------------------------------------------------------------
# Common polars expressions
# --------------------------------------------------------------------


def select(*expr: pl.Expr, **named_expr: pl.Expr) -> Transform:
    """Create a select transform from given expressions.

    Parameters
    ----------
    *expr : pl.Expr
        Positional expressions to pass to select.
    **named_expr : pl.Expr
        Keyword expressions to pass to select.

    Returns
    -------
    Transform

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

    Returns
    -------
    Transform

    """

    def _with_columns(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns(*expr, **named_expr)

    _with_columns.__name__ = "with_columns"
    _with_columns.__qualname__ = "transforms.with_columns"
    return _with_columns
