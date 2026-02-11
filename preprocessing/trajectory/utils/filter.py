from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from preprocessing.trajectory.interface import ProcessorConfig
    from preprocessing.trajectory.utils.common import T_DataFame


def filter_scene(
    data: T_DataFame,
    config: ProcessorConfig,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = "agent_class",
) -> T_DataFame:
    """Filter scenes based on configuration.

    Args:
        data: Input DataFrame containing trajectory data.
        config: ProcessorConfig object with filtering criteria.
        group_by: Optional column name to group by.
        agent_id: Column name representing the agent ID.
        frame_column: Column name representing the frame index.
        category_column: Column name representing the agent category.

    Returns:
        Filtered DataFrame based on the configuration.

    """
    filter_expr = filter_scene_expr(
        config=config,
        group_by=group_by,
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
    )
    return data.filter(filter_expr)


def filter_scene_expr(
    config: ProcessorConfig,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = "agent_class",
) -> pl.Expr:
    """Express scene filtering logic as a Polars expression.

    Args:
        config: ProcessorConfig object with filtering criteria.
        group_by: Optional column name to group by.
        agent_id: Column name representing the agent ID.
        frame_column: Column name representing the frame index.
        category_column: Column name representing the agent category.

    Returns:
        Expression for filtering scenes based on the configuration.

    """
    if config.scene_filtering is None:
        return pl.lit(value=True)

    filtering = config.scene_filtering
    group_by = [group_by] if isinstance(group_by, str) else list(group_by or [])
    scene_window = group_by or pl.lit(1)

    conditions = []

    # 0. Agent Type Filtering
    # We establish valid types early so they don't count towards min_agents
    valid_type_expr = pl.lit(value=True)
    if filtering.filter_agent_category is not None:
        if category_column is None:
            msg = "category_column must be provided when filter_agent_class is set."
            raise ValueError(msg)

        # Identify rows that are NOT in the removal list
        valid_type_expr = ~pl.col(category_column).is_in(
            filtering.filter_agent_category
        )
        conditions.append(valid_type_expr)

    # 1. Base Frame Filtering (Relative offsets check)
    if filtering.require_frames is not None:
        required_set = set(filtering.require_frames)
        n_required = len(required_set)

        relative_frame = pl.col(frame_column) - pl.col(frame_column).min().over(
            scene_window
        )

        has_all_frames = (
            relative_frame
            .filter(relative_frame.is_in(required_set))
            .n_unique()
            .over(scene_window)
            == n_required
        )
        conditions.append(has_all_frames)

    # 2. Individual Agent Validity
    # agent_validity determines which agents count towards the scene-level min_agents threshold.
    # It must exclude removed types.
    agent_validity = valid_type_expr

    if filtering.require_all_valid:
        total_len = config.input_len + config.output_len
        agent_window = [*group_by, agent_id] if group_by else [agent_id]

        # Check length AND type validity
        track_len_valid = pl.len().over(agent_window) == total_len
        agent_valid_expr = track_len_valid & valid_type_expr

        conditions.append(agent_valid_expr)
        agent_validity = agent_valid_expr

    # 3. Scene-Level: Minimum Valid Agents
    if filtering.min_agents > 0:
        # Count UNIQUE agents that are valid (passed type check and/or length check)
        valid_agent_count = (
            pl.col(agent_id).filter(agent_validity).n_unique().over(scene_window)
        )
        conditions.append(valid_agent_count >= filtering.min_agents)

    # 4. Scene-Level: Prediction Frame Existence
    if filtering.require_prediction_frame:
        start_frame = pl.col(frame_column).min().over(scene_window)
        target_frame = start_frame + config.input_len - 1

        has_pred_frame = (
            (pl.col(frame_column) == target_frame).any().over(scene_window)
        )
        conditions.append(has_pred_frame)

    if not conditions:
        return pl.lit(value=True)

    return pl.all_horizontal(conditions)
