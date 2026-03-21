from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from dronalize.pipeline.functional.basic import lazy

if TYPE_CHECKING:
    from dronalize._internal._types import DataFrameT


def rebalance_highway_agents(
    data: DataFrameT,
    ratio: float = 2.0,
    req_lane_changes: int = 1,
    *,
    agent_id: str = "id",
    lane_changes_col: str = "lane_changes",
    seed: int | None = None,
) -> DataFrameT:
    """Rebalance data to enforce a specific ratio of Lane Changing (LC) to Lane Keeping (LK) agents.

    Ratio formula: N_LC / N_LK = ratio
    Therefore: Target N_LK = N_LC / ratio

    Parameters
    ----------
    data : T_DataFrame
        Input DataFrame or LazyFrame.
    ratio : float, optional
        Target ratio of LC agents to LK agents (e.g., 2.0 means 2 LC for every 1 LK).
    req_lane_changes : int, optional
        Minimum lane changes to be considered an LC agent.
    agent_id : str, optional
        Column name for agent identifiers.
    lane_changes_col : str, optional
        Column containing lane change counts (assumed pre-calculated per agent).
    seed : int, optional
        Random seed for reproducibility of sampling. This will also perform a sort
        operation to ensure deterministic sampling across runs.

    Examples
    --------
    For a simple dataset with 6 agents, where agents 0-3 are LK (0 lane changes) and agents 4-5
    are LC (2 lane changes), with a ratio of 2.0 we keep all 2 LC agents and only 1 LK agent
    (since ratio is 2 LC : 1 LK). The resulting dataset should contain 3 agents total.

    >>> df = pl.DataFrame(
    ...     {
    ...         "id": [0, 1, 2, 3, 4, 5],
    ...         "lane_changes": [0, 0, 0, 0, 2, 2],
    ...     }
    ... )
    >>> rebalance_highway_agents(df, ratio=2.0, req_lane_changes=1, agent_id="id", seed=42)
    shape: (3, 2)
    ┌─────┬──────────────┐
    │ id  ┆ lane_changes │
    │ --- ┆ ---          │
    │ i64 ┆ i64          │
    ╞═════╪══════════════╡
    │ 3   ┆ 0            │
    │ 4   ┆ 2            │
    │ 5   ┆ 2            │
    └─────┴──────────────┘

    """
    # 1. Normalize input to LazyFrame to unify logic
    lazy_data = lazy(data)

    # 2. Extract unique agents and classify them
    # We group by ID and take the max of lane_changes to see if they EVER met the criteria
    # (assuming n_lanechange_col might be cumulative or instantaneous)
    agent_stats = (
        lazy_data
        .group_by(agent_id)
        .agg(pl.col(lane_changes_col).max().alias("max_lc"))
        .with_columns((pl.col("max_lc") >= req_lane_changes).alias("is_lc_agent"))
    )

    if seed is not None:
        # If seed is set, detmerinistic sampling is desired. Sorting by agent_id ensures the same
        # order across runs.
        agent_stats = agent_stats.sort(agent_id)

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
        sampled_lk = lk_agents.sample(n=n_lk_target, shuffle=True, seed=seed)

    # 4. Combine allowed IDs
    valid_ids = pl.concat([lc_agents.select(agent_id), sampled_lk.select(agent_id)])

    # 5. Filter the original data
    # semi_join is equivalent to SQL "WHERE id IN (...)" but more optimized for frames
    result = lazy_data.join(valid_ids.lazy(), on=agent_id, how="semi")

    # Return valid type matching input
    if isinstance(data, pl.DataFrame):
        return result.collect()
    return result
