"""Reusable composite pipeline factories for common trajectory processing.

This module provides pre-built `~dronalize.core.pipeline.Pipeline`
fragments that encapsulate recurring multi-step processing stages found
across dataset loaders.  They are designed to be composed into a loader's
`pipeline()` method via `Pipeline.compose` or the `>>` operator.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dronalize.core.transforms as tr
from dronalize.core.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.core.datatypes.loader_config import LoaderConfig

# TODO: Look over docs and update if needed


def trajectory_pipeline(
    config: LoaderConfig,
    *,
    agent_id: str = "id",
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    category_column: str | None = "agent_category",
    min_agents: int = 2,
    add_derivative: bool = True,
    add_second_derivative: bool = True,
    derivative_rename: dict[int, list[str]] | None = None,
    forward_fill: Sequence[str] | None = None,
) -> Pipeline:
    """Build the standard windowing → filtering → resampling → splitting pipeline.

    This is the canonical processing pipeline that most trajectory-prediction
    dataset loaders need.

    The returned pipeline contains the following stages (each skipped
    automatically when the corresponding config field is `None`):

    1. **Windowing** — sliding-window sampling (`config.window_params`).
    2. **Scene filtering** — agent validity / category / speed filtering
       (`config.scene_filtering`).
    3. **Post-filter validation** — require at least *min_agents* per window
       (only when windowing is active).
    4. **Resampling** — temporal resampling with optional derivatives.
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
    pos_columns : Sequence[str], optional
        Position column names.  Defaults to `("x", "y")`.
    category_column : str or None, optional
        Agent-category column name.  Defaults to `"agent_category"`.
    min_agents : int, optional
        Minimum agents per scene/window.  Defaults to 2.
    add_derivative : bool, optional
        Compute first-order derivatives.  Defaults to `True`.
    add_second_derivative : bool, optional
        Compute second-order derivatives.  Defaults to `True`.
    derivative_rename : dict[int, list[str]] or None, optional
        Custom derivative column names.
    forward_fill : Sequence[str] or None, optional
        Columns to forward-fill instead of interpolate.

    Returns
    -------
    Pipeline
        A fully composed pipeline ready to be further extended or executed.

    Examples
    --------
    Minimal usage inside a loader's `pipeline` method::

        @override
        def pipeline(self) -> Pipeline:
            config = self.loader_config
            return (
                Pipeline()
                .then(tr.require_min("id", minimum=2))
                .compose(trajectory_pipeline(config, derivative_rename=self.derivative_names()))
                .then(tr.yaw_from_vel())
                .with_columns(agent_category=pl.lit(AgentCategory.PEDESTRIAN))
            )
    """
    has_window = config.window_params is not None
    group_by_filter: str | None = "window_index" if has_window else None
    require_cols: str | list[str] = ["window_index", agent_id] if has_window else agent_id
    group_by_resample: list[str] = ["window_index", agent_id] if has_window else [agent_id]

    return (
        Pipeline()
        # 1. Windowing
        .then_if_present(
            lambda w: tr.window(w.window_size, w.step_size),
            arg=config.window_params,
        )
        # 2. Scene filtering
        .then_if_present(
            lambda c: tr.filter_scene(
                c,
                group_by=group_by_filter,
                agent_id=agent_id,
                frame_column=frame_column,
                category_column=category_column,
            ),
            arg=config.scene_filtering,
        )
        # 3. Post-filter minimum-agent validation
        .then(
            tr.require_min(require_cols, minimum=min_agents),
            when=has_window,
        )
        # 4. Resampling (+ derivatives)
        .then(
            tr.resample(
                config.resampling,
                config.sample_time,
                frame_column=frame_column,
                pos_columns=pos_columns,
                group_by=group_by_resample,
                add_derivative=add_derivative,
                add_second_derivative=add_second_derivative,
                derivative_rename=derivative_rename,
                forward_fill=forward_fill,
            )
        )
        # 5. Split by window
        .then_flat_map(tr.group_by_yield("window_index"), when=has_window)
    )
