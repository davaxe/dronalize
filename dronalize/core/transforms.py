"""Built-in transform / fan-out factory functions for pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from dronalize.common.trajectory.basic import yaw_from_pos_expr, yaw_from_vel_expr
from dronalize.common.trajectory.derivative import derivative as _derivative_impl
from dronalize.common.trajectory.filter import FilteringConfig, filter_scene_expr
from dronalize.common.trajectory.rebalance import rebalance_highway_agents
from dronalize.common.trajectory.resample import Resampling, resample_tracks
from dronalize.common.trajectory.window import sliding_window
from dronalize.core.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dronalize.core.datatypes.loader_config import LoaderConfig
    from dronalize.core.pipeline import FlatMapTransform, Transform

__all__ = [
    "derivative",
    "filter_scene",
    "group_by_yield",
    "pipeline_from_config",
    "rebalance",
    "resample",
    "window",
    "yaw_from_pos",
    "yaw_from_vel",
]


# -------------------------------------------------------------------
# Filtering
# -------------------------------------------------------------------


def filter_scene(
    config: FilteringConfig | None = None,
    *,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = "agent_category",
) -> Transform:
    """Create a filtering transform.

    Wraps :func:`~dronalize.common.trajectory.filter.filter_scene_expr`.

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


# -------------------------------------------------------------------
# Resampling
# -------------------------------------------------------------------


def resample(
    resampling: Resampling | None = None,
    sample_time: float = 1.0,
    *,
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    group_by: str | Sequence[str] | None = None,
    add_derivative: bool = False,
    add_second_derivative: bool = False,
    derivative_rename: dict[int, list[str]] | None = None,
    forward_fill: Sequence[str] | None = None,
) -> Transform:
    """Create a resampling transform.

    Wraps :func:`~dronalize.common.trajectory.resample.resample_tracks`.
    If the resampling object specifies no resampling (ratio 1:1), the transform
    is still applied (it becomes a no-op for the resampling itself but
    may still add derivatives).

    Parameters
    ----------
    resampling : Resampling, optional
        Resampling parameters.
    sample_time : float, optional
        Original sample time in seconds.
    frame_column : str, optional
        Column name for the frame index.
    pos_columns : Sequence[str], optional
        Position column names.
    group_by : str or Sequence[str], optional
        Columns to partition tracks by.
    add_derivative : bool, optional
        Compute first-order derivatives.
    add_second_derivative : bool, optional
        Compute second-order derivatives.
    derivative_rename : dict[int, list[str]], optional
        Custom names for derivative columns.
    forward_fill : Sequence[str], optional
        Columns to forward-fill instead of interpolate.

    Returns
    -------
    Transform

    """
    resampling_obj = resampling or Resampling(1, 1)

    def _resample(df: pl.LazyFrame) -> pl.LazyFrame:
        return resample_tracks(
            df,
            resampling_obj,
            frame_column=frame_column,
            pos_columns=pos_columns,
            group_by=group_by,
            add_derivative=add_derivative,
            add_second_derivative=add_second_derivative,
            dt=sample_time,
            derivative_rename=derivative_rename,
            forward_fill=forward_fill,
        )

    _resample.__name__ = "resample"
    _resample.__qualname__ = "transforms.resample"
    return _resample


# -------------------------------------------------------------------
# Derivatives
# -------------------------------------------------------------------


def derivative(
    *columns: str,
    dt: float = 1.0,
    n: int = 1,
    include_intermediate: bool = False,
    group_by: str | Sequence[str] | None = None,
    derivative_rename: dict[int, list[str]] | None = None,
) -> Transform:
    """Create a derivative transform.

    Wraps :func:`~dronalize.common.trajectory.derivative.derivative`.

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


# -------------------------------------------------------------------
# Yaw estimation
# -------------------------------------------------------------------


def yaw_from_vel(
    vx_col: str = "vx",
    vy_col: str = "vy",
    yaw_col: str = "yaw",
    *,
    only_null: bool = False,
) -> Transform:
    """Create a yaw-from-velocity transform.

    Wraps :func:`~dronalize.common.trajectory.basic.yaw_from_vel`.

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
                .then(yaw_from_vel_expr(vx_col, vy_col, yaw_col))
                .otherwise(pl.col(yaw_col))
                .alias(yaw_col)
            )

    else:

        def _yaw_from_vel(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(yaw_from_vel_expr(vx_col, vy_col, yaw_col))

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

    Wraps :func:`~dronalize.common.trajectory.basic.yaw_from_pos`.

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
                .then(yaw_from_pos_expr(x_col, y_col, yaw_col))
                .otherwise(pl.col(yaw_col))
                .alias(yaw_col)
            )

    else:

        def _yaw_from_pos(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(yaw_from_pos_expr(x_col, y_col, yaw_col))

    _yaw_from_pos.__name__ = "yaw_from_pos"
    _yaw_from_pos.__qualname__ = "transforms.yaw_from_pos"
    return _yaw_from_pos


# -------------------------------------------------------------------
# Sliding window (fan-out)
# -------------------------------------------------------------------


def window(
    window_size: int,
    step_size: int,
    *,
    sliding_col: str = "frame",
    offset_sliding_col: bool = True,
) -> FlatMapTransform:
    """Create a sliding-window fan-out transform.

    This is a 1:N step: a single LazyFrame is split into many
    overlapping windows, each yielded as a separate LazyFrame.

    Wraps :func:`~dronalize.common.trajectory.window.sliding_window`.

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
    FlatMapTransform

    """

    def _window(df: pl.LazyFrame) -> Iterable[pl.LazyFrame]:
        for window_df in sliding_window(
            df,
            window_size=window_size,
            step_size=step_size,
            sliding_col=sliding_col,
            return_iterable=True,
        ):
            if offset_sliding_col:
                window_df = window_df.with_columns(  # noqa: PLW2901
                    pl.col(sliding_col) - pl.col(sliding_col).min()
                )
            yield window_df.lazy()

    _window.__name__ = "window"
    _window.__qualname__ = "transforms.window"
    return _window


# -------------------------------------------------------------------
# Rebalancing (highway lane-change ratio)
# -------------------------------------------------------------------


def rebalance(
    ratio: float,
    *,
    req_lane_changes: int = 1,
    agent_id: str = "id",
    n_lanechange_col: str = "lane_changes",
    seed: int | None = None,
    drop_lanechange_col: bool = True,
) -> Transform:
    """Create a highway agent rebalancing transform.

    Wraps :func:`~dronalize.common.trajectory.rebalance.rebalance_highway_agents`.

    Parameters
    ----------
    ratio : float
        Target LC/LK ratio.
    req_lane_changes : int, optional
        Minimum lane changes to count as LC agent.
    agent_id : str, optional
        Agent ID column.
    n_lanechange_col : str, optional
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
            n_lanechange_col=n_lanechange_col,
            seed=seed,
        )
        if drop_lanechange_col:
            result = result.drop(n_lanechange_col)
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


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline construction helper from LoaderConfig
# ═══════════════════════════════════════════════════════════════════════════


def pipeline_from_config(
    config: LoaderConfig,
    *,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = "agent_category",
    pos_columns: Sequence[str] = ("x", "y"),
    add_derivative: bool = False,
    add_second_derivative: bool = False,
    derivative_rename: dict[int, list[str]] | None = None,
    forward_fill: Sequence[str] | None = None,
    sliding_col: str = "frame",
    offset_sliding_col: bool = True,
    add_yaw_from_vel: bool = False,
) -> Pipeline:
    """Build a standard :class:`~dronalize.core.pipeline.Pipeline` from a :class:`LoaderConfig`.

    This constructs the "canonical" processing pipeline that
    ``prepare_agent_trajectories`` used to perform monolithically:

    1. (optional) sliding window → fan-out
    2. filtering
    3. resampling + derivatives
    4. (optional) yaw from velocity

    Parameters
    ----------
    config : LoaderConfig
        Full loader configuration.
    group_by : str or Sequence[str], optional
        Scene-level grouping columns.
    agent_id : str, optional
        Agent identifier column.
    frame_column : str, optional
        Frame index column.
    category_column : str, optional
        Agent category column.
    pos_columns : Sequence[str], optional
        Position columns.
    add_derivative : bool, optional
        Add first-order derivatives.
    add_second_derivative : bool, optional
        Add second-order derivatives.
    derivative_rename : dict[int, list[str]], optional
        Custom derivative column names.
    forward_fill : Sequence[str], optional
        Columns to forward fill after resampling.
    sliding_col : str, optional
        Column for sliding window.
    offset_sliding_col : bool, optional
        Zero-offset the sliding column per window.
    add_yaw_from_vel : bool, optional
        Append a yaw-from-velocity step.

    Returns
    -------
    Pipeline
        A ready-to-use pipeline.

    """
    pipe = Pipeline()

    # Determine the full group_by list.  If windowing is used, the sliding
    # window adds a `window_index` column that must be included in
    # downstream group-by operations.
    group_by_list: list[str] = [group_by] if isinstance(group_by, str) else list(group_by or [])
    resample_group_by: list[str] = list(group_by_list)

    # 1. Sliding window (fan-out or inline)
    if config.window_params is not None:
        pipe = pipe.then(
            _inline_window(
                window_size=config.window_params.window_size,
                step_size=config.window_params.step_size,
                sliding_col=sliding_col,
            )
        )
        resample_group_by.append("window_index")

    # 2. Filtering
    pipe = pipe.then(
        filter_scene(
            config.scene_filtering,
            group_by=resample_group_by[-1] if resample_group_by else None,
            agent_id=agent_id,
            frame_column=frame_column,
            category_column=category_column,
        )
    )

    # Ensure single-point groups are dropped (same as prepare_agent_trajectories)
    resample_and_agent_groups = [*resample_group_by, agent_id]

    def _drop_singletons(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(pl.len().over(resample_and_agent_groups) > 1)

    _drop_singletons.__name__ = "drop_singletons"
    pipe = pipe.then(_drop_singletons)

    # 3. Resampling + derivatives
    resolved_forward_fill = (
        [category_column, *(forward_fill or [])] if category_column else forward_fill or None
    )

    pipe = pipe.then(
        resample(
            config.resampling,
            config.sample_time,
            frame_column=frame_column,
            pos_columns=pos_columns,
            group_by=resample_and_agent_groups or None,
            add_derivative=add_derivative,
            add_second_derivative=add_second_derivative,
            derivative_rename=derivative_rename,
            forward_fill=resolved_forward_fill,
        )
    )

    # 4. If windowed, split into individual windows
    if config.window_params is not None:
        pipe = pipe.then_flat_map(
            _split_windows(
                sliding_col=sliding_col,
                offset_sliding_col=offset_sliding_col,
            )
        )

    # 5. Optional yaw from velocity
    if add_yaw_from_vel:
        pipe = pipe.then(yaw_from_vel())

    return pipe


# ---------------------------------------------------------------------------
# Internal helpers for pipeline_from_config
# ---------------------------------------------------------------------------


def _inline_window(
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
) -> Transform:
    """Apply sliding window as a 1:1 transform that keeps ``window_index``.

    This mirrors the ``return_iterable=False`` path of
    :func:`~dronalize.common.trajectory.window.sliding_window`, keeping
    all windows in a single frame tagged with ``window_index``.
    The fan-out into individual frames happens later via
    :func:`_split_windows`.
    """

    def _apply(df: pl.LazyFrame) -> pl.LazyFrame:
        return sliding_window(
            df,
            window_size=window_size,
            step_size=step_size,
            sliding_col=sliding_col,
            is_sorted=False,
            return_iterable=False,
        )

    _apply.__name__ = "inline_window"
    _apply.__qualname__ = "transforms.inline_window"
    return _apply


def _split_windows(
    sliding_col: str = "frame",
    *,
    offset_sliding_col: bool = True,
) -> FlatMapTransform:
    """Fan-out that groups by ``window_index`` and yields each window."""

    def _apply(df: pl.LazyFrame) -> Iterable[pl.LazyFrame]:
        collected = df.collect()
        for _, group in collected.group_by("window_index"):
            if offset_sliding_col:
                group = group.with_columns(  # noqa: PLW2901
                    pl.col(sliding_col) - pl.col(sliding_col).min()
                )
            yield group.lazy().drop("window_index")

    _apply.__name__ = "split_windows"
    _apply.__qualname__ = "transforms.split_windows"
    return _apply
