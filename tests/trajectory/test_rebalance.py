import polars as pl

from dronalize.common.trajectory.rebalance import rebalance_highway_agents


def create_dummy_data(n_lc_agents: int, n_lk_agents: int) -> pl.DataFrame:
    """Create a dummy trajectory dataset.

    Each agent has 10 rows of data.
    """
    # Create Lane Changing (LC) Agents (IDs starting at 0)
    lc_ids = range(n_lc_agents)
    lc_data = {
        "id": [uid for uid in lc_ids for _ in range(10)],
        "lane_changes": [5 for _ in lc_ids for _ in range(10)],  # All have > 1 lane change
        "frame_id": [i for _ in lc_ids for i in range(10)],
    }

    # Create Lane Keeping (LK) Agents (IDs starting after LC agents)
    lk_ids = range(n_lc_agents, n_lc_agents + n_lk_agents)
    lk_data = {
        "id": [uid for uid in lk_ids for _ in range(10)],
        "lane_changes": [0 for _ in lk_ids for _ in range(10)],  # All have 0 lane changes
        "frame_id": [i for _ in lk_ids for i in range(10)],
    }

    df_lc = pl.DataFrame(lc_data)
    df_lk = pl.DataFrame(lk_data)

    return pl.concat([df_lc, df_lk])


def test_ratio_enforcement() -> None:
    """Scenario: 10 LC agents and 20 LK agents.

    Target: Ratio of 2.0 (2 LC : 1 LK).
    Expected: Keep all 10 LC agents, and downsample LK to 5 agents.
    Total Agents = 15.
    """
    df = create_dummy_data(n_lc_agents=10, n_lk_agents=20)

    result = rebalance_highway_agents(
        df,
        ratio=2.0,
        req_lane_changes=1,
        agent_id="id",
        lane_changes_col="lane_changes",
    )

    # Verify result is a DataFrame (since input was DataFrame)
    assert isinstance(result, pl.DataFrame)

    # unique agents remaining
    unique_agents = result["id"].unique()
    n_unique = len(unique_agents)

    # 10 LC + (10 / 2) LK = 15 expected agents
    assert n_unique == 15

    # Verify no LC agents were lost
    # IDs 0-9 were LC. They should all be present.
    assert result.filter(pl.col("id") < 10)["id"].n_unique() == 10


def test_lazy_frame_support() -> None:
    """Scenario: Pass a LazyFrame input.

    Expected: Return a LazyFrame output and correct filtering when collected.
    """
    df = create_dummy_data(n_lc_agents=10, n_lk_agents=10)
    lazy_df = df.lazy()

    # Ratio 1.0 -> 10 LC : 10 LK. Should keep everyone.
    result_lazy = rebalance_highway_agents(lazy_df, ratio=1.0)

    assert isinstance(result_lazy, pl.LazyFrame)

    result = result_lazy.collect()
    assert result["id"].n_unique() == 20


def test_insufficient_lk_data() -> None:
    """Scenario: Ratio requires MORE LK agents than exist.

    Setup: 10 LC, 2 LK.
    Target Ratio: 1.0 (requires 10 LK).
    Expected: Function should return all available LK agents (2) without crashing.
    Total agents = 12.
    """
    df = create_dummy_data(n_lc_agents=10, n_lk_agents=2)

    result = rebalance_highway_agents(df, ratio=1.0)

    # Should have all 10 LC and all 2 LK
    assert result["id"].n_unique() == 12


def test_custom_lane_change_threshold() -> None:
    """Scenario: Increase required lane changes to 10.

    Setup: The 'LC' dummy agents only have 5 changes.
    Expected: Everyone is classified as LK.
    Result should be empty or handle 0 LC agents gracefully depending on logic.
    (Current logic: 0 LC -> 0 target LK -> result is empty).
    """
    df = create_dummy_data(n_lc_agents=5, n_lk_agents=5)

    # Set req to 10. The agents with 5 changes are now considered LK.
    # Total LK = 10. Total LC = 0.
    result = rebalance_highway_agents(df, req_lane_changes=10)

    # 0 LC agents / Ratio 2 = 0 LK agents required.
    # Result should be empty.
    assert len(result) == 0
