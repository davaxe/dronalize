from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from preprocessing.common.trajectory_utils.filter import filter_scene_expr
from preprocessing.common.trajectory_utils.resample import resample_tracks
from preprocessing.common.trajectory_utils.window import sliding_window

if TYPE_CHECKING:
    from collections.abc import Iterable

from preprocessing.core.interface.trajectory import LoaderConfig, Resampling


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
) -> Iterable[pl.LazyFrame]:
    """Prepare agent trajectories for processing.

    This function performs several common preprocessing steps on the input scenes:
    1. Optionally applies a sliding window to the scenes if `window_params` is set in the config.
    2. Filters the scenes based on the criteria specified in the config.
    3. Resamples the agent trajectories to a common time step, and optionally adds derivatives.

    Args:
        scenes: A LazyFrame containing the raw scenes to be processed.
        config: The configuration for processing the scenes.
        add_derivative: Whether to compute and add the first derivative of the trajectories.
        add_second_derivative: Whether to compute and add the second derivative of the trajectories.
        sliding_col: The column name to use for sliding window operations.
        agent_category_col: The column name containing agent categories.
        derivative_rename: A dictionary mapping old derivative column names to new names.
        offset_sliding_col: Whether to offset the sliding column to start from zero in each window.

    Yields:
        Processed scenes as LazyFrames, ready for further processing or saving.

    """
    resampling = config.resampling or Resampling(1, 1)
    group_by: list[str] = []

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
            config,
            agent_id="id",
            group_by=group_by[-1] if len(group_by) > 0 else None,
            category_column=agent_category_col,
        )
    )
    group_by.append("id")
    scenes_filtered = scenes_filtered.filter(pl.len().over(group_by) > 1)
    scenes_resampled = resample_tracks(
        scenes_filtered,
        resampling.up,
        resampling.down,
        group_by=group_by,
        add_derivative=add_derivative,
        add_second_derivative=add_second_derivative,
        method=resampling.method,
        dt=config.sample_time,
        derivative_rename=derivative_rename,
        forward_fill=[agent_category_col] if agent_category_col else None,
    )

    if config.window_params is None:
        yield scenes_resampled.lazy()
    else:
        for _, group in scenes_resampled.collect().group_by("window_index"):
            if offset_sliding_col:
                group = group.with_columns(pl.col(sliding_col) - pl.col(sliding_col).min())  # noqa: PLW2901
            yield group.lazy().drop("window_index")
