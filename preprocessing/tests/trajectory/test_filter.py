import polars as pl
from polars.testing import assert_frame_equal

from preprocessing.core.categories import AgentCategory
from preprocessing.trajectory.interface import ProcessorConfig, SceneFiltering
from preprocessing.trajectory.utils import filter_scene


def test_no_scene_filtering() -> None:
    """Test that setting scene_filtering to None returns the original DataFrame."""
    df = pl.DataFrame({"id": [1, 1], "frame": [0, 1], "scene": [1, 1]})
    config = ProcessorConfig(
        input_len=1, output_len=1, sample_time=0.1, scene_filtering=None
    )

    result = filter_scene(df, config, group_by="scene")
    assert len(result) == 2

    assert_frame_equal(result, df)


def test_filter_agent_category() -> None:
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

    config = ProcessorConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
        scene_filtering=SceneFiltering(
            min_agents=0,
            require_prediction_frame=False,
            filter_agent_category=[AgentCategory.UNIMPORTANT],
        ),
    )

    result = filter_scene(df, config, group_by="scene", category_column="category")

    assert len(result) == 2
    assert AgentCategory.UNIMPORTANT not in result["category"].to_list()


def test_min_agents_threshold() -> None:
    """Test that scenes failing the minimum valid agent count are dropped."""
    df = pl.DataFrame({
        "scene": [1, 1, 2],  # Scene 1 has 2 agents, Scene 2 has 1 agent
        "id": [1, 2, 3],
        "frame": [0, 0, 0],
    })

    config = ProcessorConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
        scene_filtering=SceneFiltering(min_agents=2, require_prediction_frame=False),
    )

    result = filter_scene(df, config, group_by="scene")

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
    config_strict = ProcessorConfig(
        input_len=2,
        output_len=2,
        sample_time=0.1,
        scene_filtering=SceneFiltering(
            min_agents=2, require_all_valid=True, require_prediction_frame=False
        ),
    )
    result_strict = filter_scene(df, config_strict, group_by="scene")
    assert len(result_strict) == 0

    # Case B: require 1 valid agent. Agent 1 is valid, so the scene survives,
    # but the invalid agent 2 should be filtered out.
    config_lenient = ProcessorConfig(
        input_len=2,
        output_len=2,
        sample_time=0.1,
        scene_filtering=SceneFiltering(
            min_agents=1, require_all_valid=True, require_prediction_frame=False
        ),
    )
    result_lenient = filter_scene(df, config_lenient, group_by="scene")
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

    config = ProcessorConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
        scene_filtering=SceneFiltering(
            min_agents=0,
            require_prediction_frame=False,
            require_frames=[0, 4],  # We require relative frames 0 and 4
        ),
    )

    result = filter_scene(df, config, group_by="scene")

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

    config = ProcessorConfig(
        input_len=3,
        output_len=1,
        sample_time=0.1,
        scene_filtering=SceneFiltering(min_agents=0, require_prediction_frame=True),
    )

    result = filter_scene(df, config, group_by="scene")

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

    # We have 4 agents total, but 2 are UNIMPORTANT.
    # If we require min_agents=3, the 2 valid ones won't meet the threshold.
    config = ProcessorConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
        scene_filtering=SceneFiltering(
            min_agents=3,
            require_prediction_frame=False,
            filter_agent_category=[AgentCategory.UNIMPORTANT],
        ),
    )

    result = filter_scene(df, config, group_by="scene", category_column="category")

    assert len(result) == 0


def test_no_groupby() -> None:
    """Test functionality when no group_by column is provided (entire df is one scene)."""
    df = pl.DataFrame({"id": [1, 2], "frame": [0, 0]})

    config = ProcessorConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
        scene_filtering=SceneFiltering(min_agents=2, require_prediction_frame=False),
    )

    # Because group_by is None, it groups over pl.lit(1)
    result = filter_scene(df, config, group_by=None)

    assert len(result) == 2
