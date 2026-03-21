import polars as pl
from polars.testing import assert_frame_equal

from dronalize.categories import AgentCategory
from dronalize.config import LoaderConfig
from dronalize.config.filtering import FilteringConfig
from dronalize.pipeline.functional.filter import filter_scene


def test_no_scene_filtering() -> None:
    """Test that setting scene_filtering to None returns the original DataFrame."""
    df = pl.DataFrame({"id": [1, 1], "frame": [0, 1], "scene": [1, 1]})
    config = LoaderConfig(input_len=1, output_len=1, sample_time=0.1, filtering=None)

    result = filter_scene(df, config.filtering, group_by="scene")
    assert len(result) == 2

    assert_frame_equal(result, df)


def test_exclude_agent_categories() -> None:
    """Test that specific agent categories are filtered out correctly."""
    df = pl.DataFrame({
        "scene": [1, 1, 1],
        "id": [1, 2, 3],
        "frame": [0, 0, 0],
        "category": [
            AgentCategory.CAR,
            AgentCategory.PEDESTRIAN,
            AgentCategory.UNIMPORTANT,
        ],
    })

    config = LoaderConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
        filtering=FilteringConfig.create(
            min_agents=0,
            exclude_agent_categories=[AgentCategory.UNIMPORTANT],
        ),
    )

    result = filter_scene(df, config.filtering, group_by="scene", category_column="category")

    assert len(result) == 2
    assert AgentCategory.UNIMPORTANT not in result["category"].to_list()


def test_min_agents_threshold() -> None:
    """Test that scenes failing the minimum valid agent count are dropped."""
    df = pl.DataFrame({
        "scene": [1, 1, 2],  # Scene 1 has 2 agents, Scene 2 has 1 agent
        "id": [1, 2, 3],
        "frame": [0, 0, 0],
    })

    config = LoaderConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
        filtering=FilteringConfig(min_agents=2),
    )

    result = filter_scene(df, config.filtering, group_by="scene")

    # Scene 2 should be completely removed
    assert result["scene"].unique().to_list() == [1]
    assert len(result) == 2


def test_require_all_valid() -> None:
    """Test length validation per agent, and its impact on scene min_agents."""
    # input_len=2 + output_len=2 = 4 required frames
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1],
        # Agent 1 has 4 frames (valid), Agent 2 has 3 (invalid)
        "id": [1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 0, 1, 2],
    })

    # Case A: require 2 valid agents. Since agent 2 is invalid, scene 1 only
    # has 1 valid agent. The entire scene should drop.
    config_strict = LoaderConfig(
        input_len=2,
        output_len=2,
        sample_time=0.1,
        filtering=FilteringConfig(min_agents=2, require_all_valid=True),
    )
    result_strict = filter_scene(df, config_strict.filtering, group_by="scene")
    assert len(result_strict) == 0

    # Case B: require 1 valid agent. Agent 1 is valid, so the scene survives,
    # but the invalid agent 2 should be filtered out.
    config_lenient = LoaderConfig(
        input_len=2,
        output_len=2,
        sample_time=0.1,
        filtering=FilteringConfig(min_agents=1, require_all_valid=True),
    )
    result_lenient = filter_scene(df, config_lenient.filtering, group_by="scene")
    assert len(result_lenient) == 4
    assert result_lenient["id"].unique().to_list() == [1]


def test_require_frames() -> None:
    """Test that scenes are kept only if specific relative frame offsets exist."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 2, 2],
        "id": [1, 1, 1, 2, 2],
        "frame": [10, 12, 14, 20, 22],
        # Scene 1 relative frames: 0, 2, 4
        # Scene 2 relative frames: 0, 2
    })

    config = LoaderConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
        filtering=FilteringConfig.create(min_agents=0, require_frames=[0, 4]),
    )

    result = filter_scene(df, config.filtering, group_by="scene")

    # Scene 2 lacks relative frame 4, so it drops
    assert result["scene"].unique().to_list() == [1]


def test_require_prediction_frame() -> None:
    """Test checking for the specific target frame (start + input_len - 1)."""
    # input_len=3 means target frame offset is (3 - 1) = 2.
    df = pl.DataFrame({
        "scene": [1, 1, 1, 2, 2],
        "id": [1, 1, 1, 2, 2],
        # Scene 1 has relative frame 2. Scene 2 does not.
        "frame": [0, 1, 2, 10, 11],
    })

    config = LoaderConfig(
        input_len=3,
        output_len=1,
        sample_time=0.1,
        filtering=FilteringConfig.create(min_agents=0, require_frames=[2]),
    )

    result = filter_scene(df, config.filtering, group_by="scene")

    assert result["scene"].unique().to_list() == [1]


def test_complex_interaction() -> None:
    """Test valid type filtering interacting with min_agents threshold."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1],
        "id": [1, 2, 3, 4],
        "frame": [0, 0, 0, 0],
        "category": [
            AgentCategory.CAR,
            AgentCategory.PEDESTRIAN,
            AgentCategory.UNIMPORTANT,
            AgentCategory.UNIMPORTANT,
        ],
    })

    config = LoaderConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
        filtering=FilteringConfig.create(
            min_agents=3,
            exclude_agent_categories=[AgentCategory.UNIMPORTANT],
        ),
    )

    result = filter_scene(df, config.filtering, group_by="scene", category_column="category")

    assert len(result) == 0


def test_no_groupby() -> None:
    """Test functionality when no group_by column is provided (entire df is one scene)."""
    df = pl.DataFrame({"id": [1, 2], "frame": [0, 0]})

    config = LoaderConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
        filtering=FilteringConfig(min_agents=2),
    )

    # Because group_by is None, it groups over pl.lit(1)
    result = filter_scene(df, config.filtering, group_by=None)

    assert len(result) == 2


def test_filter_slow_agents() -> None:
    """Test that agents with an average speed below the threshold are filtered out."""
    # Agent 1: Moves 4.0 meters over 0.2 seconds (avg speed = 20 m/s)
    # Agent 2: Moves 0.2 meters over 0.2 seconds (avg speed = 1 m/s)
    # Agent 3: Stationary (avg speed = 0 m/s)
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "frame": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        "x": [0.0, 2.0, 4.0, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })

    config = LoaderConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
        filtering=FilteringConfig(
            min_agents=0,
            filter_slow_agents=2.0 * 0.1,
        ),
    )

    result = filter_scene(df, config.filtering, group_by="scene")

    # Only Agent 1 meets the speed requirement
    assert len(result) == 3
    assert result["id"].unique().to_list() == [1]


def test_min_samples_per_agent_basic() -> None:
    """Test that agents with fewer samples than the threshold are removed."""
    # Agent 1: 5 samples, Agent 2: 2 samples, Agent 3: 1 sample
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 1, 2, 2, 3],
        "frame": [0, 1, 2, 3, 4, 0, 1, 0],
    })

    config = FilteringConfig(min_agents=0, min_samples_per_agent=3)
    result = filter_scene(df, config, group_by="scene")

    # Only Agent 1 has >= 3 samples
    assert len(result) == 5
    assert result["id"].unique().to_list() == [1]


def test_min_samples_per_agent_all_pass() -> None:
    """Test that no agents are removed when all meet the threshold."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 0, 1, 2],
    })

    config = FilteringConfig(min_agents=0, min_samples_per_agent=3)
    result = filter_scene(df, config, group_by="scene")

    assert_frame_equal(result, df)


def test_min_samples_per_agent_interacts_with_min_agents() -> None:
    """Test that agents dropped by min_samples don't count toward min_agents."""
    # Scene 1: Agent 1 has 5 samples, Agent 2 has 1 sample
    # After min_samples_per_agent=3, only Agent 1 survives → 1 valid agent.
    # With min_agents=2 the entire scene should be dropped.
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 1, 2],
        "frame": [0, 1, 2, 3, 4, 0],
    })

    config = FilteringConfig(min_agents=2, min_samples_per_agent=3)
    result = filter_scene(df, config, group_by="scene")

    assert len(result) == 0


def test_min_samples_per_agent_no_groupby() -> None:
    """Test min_samples_per_agent without a group_by column."""
    df = pl.DataFrame({
        "id": [1, 1, 1, 2],
        "frame": [0, 1, 2, 0],
    })

    config = FilteringConfig(min_agents=0, min_samples_per_agent=2)
    result = filter_scene(df, config, group_by=None)

    assert len(result) == 3
    assert result["id"].unique().to_list() == [1]


def test_min_samples_per_agent_via_loader_config() -> None:
    """Test that min_samples_per_agent is correctly threaded through LoaderConfig."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1],
        "id": [1, 1, 1, 2, 2],
        "frame": [0, 1, 2, 0, 1],
    })

    config = LoaderConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
    ).with_filtering(min_agents=0, min_samples_per_agent=3)

    assert config.filtering is not None
    assert config.filtering.min_samples_per_agent == 3

    result = filter_scene(df, config.filtering, group_by="scene")

    # Only Agent 1 has >= 3 samples
    assert len(result) == 3
    assert result["id"].unique().to_list() == [1]


def test_min_samples_per_agent_none_disables() -> None:
    """Test that min_samples_per_agent=None (default) does not filter anything."""
    df = pl.DataFrame({
        "scene": [1, 1],
        "id": [1, 2],
        "frame": [0, 0],
    })

    config = FilteringConfig(min_agents=0, min_samples_per_agent=None)
    result = filter_scene(df, config, group_by="scene")

    assert_frame_equal(result, df)


def test_min_samples_per_agent_with_multiple_scenes() -> None:
    """Test that min_samples_per_agent is evaluated per scene independently."""
    # Scene 1: Agent 1 has 3 samples, Agent 2 has 1 sample
    # Scene 2: Agent 3 has 2 samples, Agent 4 has 4 samples
    # With min_samples_per_agent=3: Agent 1 (3) and Agent 4 (4) survive.
    # Scene 1 keeps Agent 1 (3 rows), Scene 2 keeps Agent 4 (4 rows).
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "id": [1, 1, 1, 2, 3, 3, 4, 4, 4, 4],
        "frame": [0, 1, 2, 0, 0, 1, 0, 1, 2, 3],
    })

    config = FilteringConfig(min_agents=1, min_samples_per_agent=3)
    result = filter_scene(df, config, group_by="scene")

    assert sorted(result["scene"].unique().to_list()) == [1, 2]
    assert sorted(result["id"].unique().to_list()) == [1, 4]
    assert len(result) == 7
