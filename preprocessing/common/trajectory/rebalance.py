from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from preprocessing.common.trajectory import T_DataFrame


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
