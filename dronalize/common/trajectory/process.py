from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from dronalize.common.trajectory.filter import filter_scene_expr
from dronalize.common.trajectory.resample import Resampling, resample_tracks
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

    Yields
    ------
    pl.LazyFrame
        Processed scenes as LazyFrames, ready for further processing or saving.

    """
    resampling = config.resampling or Resampling(1, 1)
    group_by: list[str] = []
    forward_fill = forward_fill or []

    if config.window_params is not None:
        scenes = sliding_window(
            scenes,
            window_size=config.window_params.window_size,
            step_size=config.window_params.step_size,
            sliding_col=sliding_col,
            is_sorted=False,
            return_iterable=False,
        )
        group_by.append("window_index")

    scenes_filtered = scenes.filter(
        filter_scene_expr(
            *config.filter_args(),
            agent_id="id",
            group_by=group_by[-1] if len(group_by) > 0 else None,
            category_column=agent_category_col,
        )
    )
    group_by.append("id")
    scenes_filtered = scenes_filtered.filter(pl.len().over(group_by) > 1)
    scenes_resampled = resample_tracks(
        scenes_filtered,
        resampling,
        group_by=group_by,
        add_derivative=add_derivative,
        add_second_derivative=add_second_derivative,
        dt=config.sample_time,
        derivative_rename=derivative_rename,
        forward_fill=[agent_category_col, *forward_fill]
        if agent_category_col
        else (forward_fill or None),
    )

    if config.window_params is None:
        yield scenes_resampled
    else:
        for _, group in scenes_resampled.collect().group_by("window_index"):
            if offset_sliding_col:
                group = group.with_columns(pl.col(sliding_col) - pl.col(sliding_col).min())  # noqa: PLW2901
            yield group.lazy().drop("window_index")
