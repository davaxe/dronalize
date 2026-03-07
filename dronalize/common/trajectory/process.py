from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from dronalize.common.trajectory.basic import collect
from dronalize.common.trajectory.filter import filter_scene_expr
from dronalize.common.trajectory.resample import Resampling, resample
from dronalize.common.trajectory.window import sliding_window

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.protocols.loader import LoaderConfig


def prepare_agent_trajectories(
    scenes: pl.LazyFrame,
    config: LoaderConfig,
    *,
    add_derivative: bool = False,
    add_second_derivative: bool = False,
    sliding_col: str = "frame",
    agent_category_col: str | None = "agent_category",
    derivative_rename: dict[int, list[str]] | None = None,
    offset_sliding_col: bool = True,
    forward_fill: list[str] | None = None,
    stream_windows: bool = False,
) -> Iterable[pl.LazyFrame]:
    """Prepare agent trajectories for processing.

    This function performs several common preprocessing steps on the input scenes:

    1. Optionally applies a sliding window to the scenes if `window_params` is set in the config.
    2. Filters the scenes based on the criteria specified in the config.
    3. Resamples the agent trajectories to a common time step, and optionally adds derivatives.

    Parameters
    ----------
    scenes : pl.LazyFrame
        A LazyFrame containing the raw scenes to be processed.
    config : LoaderConfig
        The configuration for processing the scenes.
    add_derivative : bool, optional
        Whether to compute and add the first derivative of the trajectories.
    add_second_derivative : bool, optional
        Whether to compute and add the second derivative of the trajectories.
    sliding_col : str, optional
        The column name to use for sliding window operations.
    agent_category_col : str, optional
        The column name containing agent categories.
    derivative_rename : dict[int, list[str]], optional
        A dictionary mapping derivative order to a list of new column names.
    offset_sliding_col : bool, optional
        Whether to offset the sliding column to start from zero in each window.
    forward_fill : list[str], optional
        List of columns to forward-fill after resampling.
    stream_windows : bool, optional
        When True, windows are processed one at a time in a streaming fashion,
        keeping only a single window in memory. This is significantly more
        memory-efficient for large source files but may be slower for smaller
        datasets because filtering and resampling are applied per window.
        When False (the default), all windows are materialized in a single
        DataFrame before being split, which is faster but requires more memory.

    Yields
    ------
    pl.LazyFrame
        Processed scenes as LazyFrames, ready for further processing or saving.

    """
    resampling = config.resampling or Resampling(1, 1)
    forward_fill = forward_fill or []

    filter_resample_kwargs = {
        "config": config,
        "resampling": resampling,
        "add_derivative": add_derivative,
        "add_second_derivative": add_second_derivative,
        "agent_category_col": agent_category_col,
        "derivative_rename": derivative_rename,
        "forward_fill": forward_fill,
    }

    if config.window_params is None:
        yield _filter_and_resample(scenes, group_by=[], **filter_resample_kwargs)
    elif stream_windows:
        yield from _windowed_streaming(
            scenes,
            sliding_col=sliding_col,
            offset_sliding_col=offset_sliding_col,
            **filter_resample_kwargs,
        )
    else:
        yield from _windowed_batch(
            scenes,
            sliding_col=sliding_col,
            offset_sliding_col=offset_sliding_col,
            **filter_resample_kwargs,
        )


def _windowed_batch(
    scenes: pl.LazyFrame,
    *,
    config: LoaderConfig,
    resampling: Resampling,
    sliding_col: str,
    offset_sliding_col: bool,
    add_derivative: bool,
    add_second_derivative: bool,
    agent_category_col: str | None,
    derivative_rename: dict[int, list[str]] | None,
    forward_fill: list[str],
) -> Iterable[pl.LazyFrame]:
    """Process all windows in a single batch (fast, higher memory).

    All windows are exploded into one DataFrame via the non-iterable sliding
    window path, then filtering and resampling are applied once across the whole
    frame. The result is split by ``window_index`` afterwards.
    """
    assert config.window_params is not None

    scenes = sliding_window(
        scenes,
        window_size=config.window_params.window_size,
        step_size=config.window_params.step_size,
        sliding_col=sliding_col,
        is_sorted=False,
        return_iterable=False,
    )

    scenes = _filter_and_resample(
        scenes,
        config=config,
        resampling=resampling,
        group_by=["window_index"],
        add_derivative=add_derivative,
        add_second_derivative=add_second_derivative,
        agent_category_col=agent_category_col,
        derivative_rename=derivative_rename,
        forward_fill=forward_fill,
    )

    for _, group in scenes.collect().group_by("window_index"):
        if offset_sliding_col:
            group = group.with_columns(pl.col(sliding_col) - pl.col(sliding_col).min())  # noqa: PLW2901
        yield group.lazy().drop("window_index")


def _windowed_streaming(
    scenes: pl.LazyFrame,
    *,
    config: LoaderConfig,
    resampling: Resampling,
    sliding_col: str,
    offset_sliding_col: bool,
    add_derivative: bool,
    add_second_derivative: bool,
    agent_category_col: str | None,
    derivative_rename: dict[int, list[str]] | None,
    forward_fill: list[str],
) -> Iterable[pl.LazyFrame]:
    """Process windows one at a time (memory-efficient, slower).

    The base data is collected once, then the sliding window iterator yields one
    ``pl.DataFrame`` per window. Filtering and resampling are applied
    independently to each window so that only a single window needs to reside in
    memory at any given time.
    """
    assert config.window_params is not None

    collected = collect(scenes)

    for window_df in sliding_window(
        collected,
        window_size=config.window_params.window_size,
        step_size=config.window_params.step_size,
        sliding_col=sliding_col,
        is_sorted=False,
        return_iterable=True,
    ):
        if offset_sliding_col:
            window_df = window_df.with_columns(  # noqa: PLW2901
                pl.col(sliding_col) - pl.col(sliding_col).min(),
            )

        yield _filter_and_resample(
            window_df.lazy(),
            config=config,
            resampling=resampling,
            group_by=[],
            add_derivative=add_derivative,
            add_second_derivative=add_second_derivative,
            agent_category_col=agent_category_col,
            derivative_rename=derivative_rename,
            forward_fill=forward_fill,
        )


def _filter_and_resample(
    scenes: pl.LazyFrame,
    config: LoaderConfig,
    resampling: Resampling,
    *,
    group_by: list[str],
    add_derivative: bool,
    add_second_derivative: bool,
    agent_category_col: str | None,
    derivative_rename: dict[int, list[str]] | None,
    forward_fill: list[str],
) -> pl.LazyFrame:
    """Apply filtering and resampling to a (possibly windowed) LazyFrame.

    Parameters
    ----------
    scenes : pl.LazyFrame
        The (possibly windowed) data to process.
    config : LoaderConfig
        Loader configuration.
    resampling : Resampling
        Resampling parameters.
    group_by : list[str]
        Additional group-by columns (e.g. ``["window_index"]``).
    add_derivative : bool
        Whether to add first derivatives.
    add_second_derivative : bool
        Whether to add second derivatives.
    agent_category_col : str or None
        Agent category column name.
    derivative_rename : dict or None
        Derivative column rename mapping.
    forward_fill : list[str]
        Columns to forward-fill after resampling.

    Returns
    -------
    pl.LazyFrame
        Filtered and resampled data.

    """
    scenes = scenes.filter(
        filter_scene_expr(
            config.scene_filtering,
            agent_id="id",
            group_by=group_by[-1] if len(group_by) > 0 else None,
            category_column=agent_category_col,
        ),
    )

    resample_group_by = [*group_by, "id"]
    return resample(
        scenes,
        resampling,
        group_by=resample_group_by,
        add_derivative=add_derivative,
        add_second_derivative=add_second_derivative,
        dt=config.sample_time,
        derivative_rename=derivative_rename,
        forward_fill=[agent_category_col, *forward_fill]
        if agent_category_col
        else (forward_fill or None),
    )
