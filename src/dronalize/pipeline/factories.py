"""Reusable composite pipeline factories for common trajectory processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dronalize.pipeline.transforms as tr
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from dronalize.config.loader import LoaderConfig


def trajectory_pipeline(
    config: LoaderConfig,
    *,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = "agent_category",
) -> Pipeline:
    """Build the standard windowing → filtering → resampling → splitting pipeline.

    This is the canonical processing pipeline that most trajectory-prediction
    dataset loaders need.

    The returned pipeline contains the following stages (each skipped
    automatically when the corresponding config field is `None`):

    1. **Windowing** — sliding-window sampling (`config.window`).
    2. **Scene filtering** — agent validity / category / speed filtering
       (`config.filtering`).
    3. **Post-filter validation** — require at least *min_agents* per window
       (only when windowing is active).
    4. **Resampling** — temporal resampling using linear interpolation for
       positions and zero-order hold for the remaining trajectory attributes.
    5. **Window splitting** — flat-map by `window_index` so downstream
       steps see one scene per frame (only when windowing is active).

    Parameters
    ----------
    config : LoaderConfig
        Full loader configuration.
    agent_id : str, optional
        Agent identifier column.  Defaults to `"id"`.
    frame_column : str, optional
        Frame / timestep column.  Defaults to `"frame"`.
    category_column : str or None, optional
        Agent-category column name.  Defaults to `"agent_category"`.

    Returns
    -------
    Pipeline
        A fully composed pipeline ready to be further extended or executed.

    """
    has_window = config.window is not None
    group_by_filter: str | None = "window_index" if has_window else None
    group_by_resample: list[str] = ["window_index", agent_id] if has_window else [agent_id]
    pipeline = Pipeline()
    if config.window is not None:
        pipeline = pipeline.then(tr.window(config.window.window_size, config.window.step_size))

    if config.filtering is not None:
        pipeline = pipeline.then(
            tr.filter_scene(
                config.filtering,
                group_by=group_by_filter,
                agent_id=agent_id,
                frame_column=frame_column,
                category_column=category_column,
            ),
        )

    pipeline = pipeline.then(
        tr.resample(
            spec=config.resampling,
            frame_column=frame_column,
            group_by=group_by_resample,
        ),
    )

    if has_window:
        pipeline = pipeline.then_flat_map(tr.group_by_yield("window_index"))

    return pipeline
