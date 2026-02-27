from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from preprocessing.common.trajectory import T_DataFrame
    from preprocessing.core.protocols.loader import LoaderConfig


# TODO: Possibly add filter option based on minimum required valid frames in
# input and output, respectively.


def filter_scene(
    data: T_DataFrame,
    config: LoaderConfig,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = None,
) -> T_DataFrame:
    """Filter scenes based on configuration.

    Args:
        data: Input DataFrame containing trajectory data.
        config: ProcessorConfig object with filtering criteria.
        group_by: Optional column name to group by. Each group will be consider an independent scene.
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
    config: LoaderConfig,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = None,
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

    # Define Windows
    scene_window = group_by or pl.lit(1)
    agent_window = [*group_by, agent_id] if group_by else [agent_id]

    conditions: list[pl.Expr] = []

    # --- 1. Global Scene Validations ---

    # Calculate Scene Start (Reference for relative calculations)
    scene_start_frame = pl.col(frame_column).min().over(scene_window)

    # Base Frame Filtering (Does the scene contain the required relative timestamps?)
    if filtering.require_frames is not None:
        required_set = set(filtering.require_frames)
        n_required = len(required_set)

        # Calculate offsets relative to scene start
        relative_frame = pl.col(frame_column) - scene_start_frame

        has_all_frames = (
            relative_frame.filter(relative_frame.is_in(required_set)).n_unique().over(scene_window)
            == n_required
        )
        conditions.append(has_all_frames)

    # --- 2. Build Agent Validity Mask ---
    # We combine ALL reasons an agent might be invalid into one mask.

    # A. Agent Type Filtering
    agent_validity = pl.lit(value=True)
    if filtering.filter_agent_category is not None and category_column is not None:
        agent_validity &= ~pl.col(category_column).is_in(filtering.filter_agent_category)

    # B. Input Frame Overlap (Condition 5)
    # Agents must appear at or before the last input frame
    last_input_frame = scene_start_frame + config.input_len - 1
    agent_start_frame = pl.col(frame_column).min().over(agent_window)
    has_input_overlap = agent_start_frame <= last_input_frame

    agent_validity &= has_input_overlap

    # C. Prediction Frame Existence
    if filtering.require_prediction_frame:
        # Target frame is calculated globally (Scene Start + Input Offset)
        target_frame = scene_start_frame + config.input_len - 1

        # Check locally: Does THIS agent have a row at that frame?
        has_pred_frame = (pl.col(frame_column) == target_frame).any().over(agent_window)

        agent_validity &= has_pred_frame

    # D. Full Length Validity
    if filtering.require_all_valid:
        total_len = config.input_len + config.output_len
        track_len_valid = pl.len().over(agent_window) == total_len

        agent_validity &= track_len_valid

    # Apply the combined mask to filter out bad agent rows
    conditions.append(agent_validity)

    # --- 3. Scene-Level: Minimum Valid Agents ---
    # Now we count only the agents that survived the "agent_validity" checks above.
    if filtering.min_agents > 0:
        valid_agent_count = pl.col(agent_id).filter(agent_validity).n_unique().over(scene_window)
        conditions.append(valid_agent_count >= filtering.min_agents)

    if not conditions:
        return pl.lit(value=True)

    return pl.all_horizontal(conditions)


def rebalance_highway_agents(
    data: T_DataFrame,
    ratio: float = 2.0,
    req_lane_changes: int = 1,
    agent_id: str = "id",
    n_lanechange_col: str = "lane_changes",
) -> T_DataFrame:
    """Rebalances data to enforce a specific ratio of Lane Changing (LC) agents to Lane Keeping (LK) agents.

    Ratio formula: N_LC / N_LK = ratio
    Therefore: Target N_LK = N_LC / ratio

    Args:
        data: Input DataFrame or LazyFrame.
        ratio: Target ratio of LC agents to LK agents (e.g., 2.0 means 2 LC for every 1 LK).
        req_lane_changes: Minimum lane changes to be considered an LC agent.
        agent_id: Column name for agent identifiers.
        n_lanechange_col: Column containing lane change counts (assumed pre-calculated per agent).

    """
    # 1. Normalize input to LazyFrame to unify logic
    lazy_data = data.lazy() if isinstance(data, pl.DataFrame) else data

    # 2. Extract unique agents and classify them
    # We group by ID and take the max of lane_changes to see if they EVER met the criteria
    # (assuming n_lanechange_col might be cumulative or instantaneous)
    agent_stats = (
        lazy_data
        .group_by(agent_id)
        .agg(pl.col(n_lanechange_col).max().alias("max_lc"))
        .with_columns((pl.col("max_lc") >= req_lane_changes).alias("is_lc_agent"))
    )

    # 3. Collect only the ID map to perform the split/sampling in memory.
    # This is efficient because we are only materializing IDs, not the full dataset.
    agent_map = agent_stats.collect()

    lc_agents = agent_map.filter(pl.col("is_lc_agent"))
    lk_agents = agent_map.filter(~pl.col("is_lc_agent"))

    n_lc = len(lc_agents)

    # Calculate target number of LK agents
    # If ratio is 2 (2 LC : 1 LK), then n_lk = n_lc / 2
    if ratio <= 0:
        msg = "Ratio must be positive."
        raise ValueError(msg)

    n_lk_target = int(n_lc / ratio)

    # Handle edge case where we don't have enough LK agents to downsample
    if n_lk_target > len(lk_agents):
        # Depending on preference, one might warn here.
        # For now, we keep all available LK agents (undersampling not possible).
        sampled_lk = lk_agents
    else:
        sampled_lk = lk_agents.sample(n=n_lk_target, shuffle=True)

    # 4. Combine allowed IDs
    valid_ids = pl.concat([lc_agents.select(agent_id), sampled_lk.select(agent_id)])

    # 5. Filter the original data
    # semi_join is equivalent to SQL "WHERE id IN (...)" but more optimized for frames
    result = lazy_data.join(valid_ids.lazy(), on=agent_id, how="semi")

    # Return valid type matching input
    if isinstance(data, pl.DataFrame):
        return result.collect()
    return result
