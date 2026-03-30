import polars as pl
from polars.testing import assert_frame_equal

from dronalize.core.categories import AgentCategory
from dronalize.processing.filters import AgentSelector, Filter, agent, cleanup, filter_scene, scene
from dronalize.processing.ingest import LoaderConfig


def test_no_filter_keeps_input() -> None:
    """Filtering disabled should return the original frame unchanged."""
    df = pl.DataFrame({"id": [1, 1], "frame": [0, 1], "scene": [1, 1]})
    config = LoaderConfig(input_len=1, output_len=1, sample_time=0.1, filter=None)

    filtered = filter_scene(df, config.filter, group_by="scene")
    diagnosed = filter_scene(df, config.filter, group_by="scene", mode="diagnose")

    assert_frame_equal(filtered, df)
    assert_frame_equal(diagnosed, df)


def test_cleanup_excludes_rows() -> None:
    """Cleanup rules should remove explicit categories while preserving the cleaned scene."""
    df = pl.DataFrame({
        "scene": [1, 1, 1],
        "id": [1, 2, 3],
        "frame": [0, 0, 0],
        "category": [AgentCategory.CAR, AgentCategory.PEDESTRIAN, AgentCategory.UNIMPORTANT],
    })

    scene_filter = Filter.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=[AgentCategory.UNIMPORTANT])],
        scene_rules=[scene.MinimumAgents(minimum=2)],
    )

    result = filter_scene(df, scene_filter, group_by="scene", category_column="category")

    assert len(result) == 2
    assert AgentCategory.UNIMPORTANT not in result["category"].to_list()
    assert sorted(result["id"].unique().to_list()) == [1, 2]


def test_cleanup_runs_before_min_agents() -> None:
    """Scenes with too few retained agents after cleanup should be dropped."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 2],
        "id": [1, 2, 3, 4],
        "frame": [0, 0, 0, 0],
        "category": [
            AgentCategory.CAR,
            AgentCategory.PEDESTRIAN,
            AgentCategory.UNIMPORTANT,
            AgentCategory.CAR,
        ],
    })

    scene_filter = Filter.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=[AgentCategory.UNIMPORTANT])],
        scene_rules=[scene.MinimumAgents(minimum=2)],
    )

    result = filter_scene(df, scene_filter, group_by="scene", category_column="category")

    assert result["scene"].unique().to_list() == [1]
    assert sorted(result["id"].unique().to_list()) == [1, 2]


def test_filtered_mode_drops_invalid() -> None:
    """Filtered mode should drop failing scenes and remove diagnostic columns."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 2, 2, 2, 2],
        "id": [1, 1, 1, 2, 3, 3, 3, 3],
        "frame": [0, 1, 2, 0, 0, 1, 2, 3],
    })

    scene_filter = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=1)],
        agent_rules=[agent.MaxMissingFrames(maximum=0)],
    )
    result = filter_scene(df, scene_filter, group_by="scene", mode="filtered")

    assert result["scene"].unique().to_list() == [2]
    assert result["id"].unique().to_list() == [3]
    assert all(not column.startswith("_filter_") for column in result.columns)


def test_diagnose_mode_keeps_rows() -> None:
    """Diagnose mode should keep all cleaned rows while annotating scene validity."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 2, 2, 2, 2],
        "id": [1, 1, 1, 2, 3, 3, 3, 3],
        "frame": [0, 1, 2, 0, 0, 1, 2, 3],
    })

    scene_filter = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=1)],
        agent_rules=[agent.MaxMissingFrames(maximum=0)],
    )
    result = filter_scene(df, scene_filter, group_by="scene", mode="diagnose")

    assert result.shape[0] == df.shape[0]
    assert result["_filter_scene_is_valid"].to_list() == ([False] * 4) + ([True] * 4)


def test_diagnose_mode_scene_columns() -> None:
    """Scene-level rules should contribute their scene-pass diagnostics."""
    df = pl.DataFrame({"scene": [1, 1, 1, 1], "id": [1, 1, 1, 2], "frame": [10, 12, 14, 10]})

    scene_filter = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0), scene.RequireFrames.define(frames=[0, 4])]
    )
    result = filter_scene(df, scene_filter, group_by="scene", mode="diagnose")

    assert result["_filter_rule_scene_min_agents_scene_passes"].to_list() == [True] * len(df)
    assert result["_filter_rule_scene_frames_scene_passes"].to_list() == [True] * len(df)
    assert result["_filter_scene_is_valid"].to_list() == [True] * len(df)


def test_diagnose_mode_uses_rule_ids() -> None:
    """Explicit rule ids should disambiguate diagnostics for same-type rules."""
    df = pl.DataFrame({
        "scene": [1, 1, 1],
        "id": [1, 2, 3],
        "frame": [0, 0, 0],
        "category": [AgentCategory.CAR, AgentCategory.PEDESTRIAN, AgentCategory.PEDESTRIAN],
    })

    scene_filter = Filter.define(
        scene_rules=[
            scene.MinimumAgents(
                minimum=1, rule_id="min_cars", selector=AgentSelector.include(["CAR"])
            ),
            scene.MinimumAgents(
                minimum=3, rule_id="min_pedestrians", selector=AgentSelector.include(["PEDESTRIAN"])
            ),
        ]
    )

    result = filter_scene(
        df, scene_filter, group_by="scene", category_column="category", mode="diagnose"
    )

    assert result["_filter_rule_min_cars_scene_passes"].to_list() == [True] * len(df)
    assert result["_filter_rule_min_pedestrians_scene_passes"].to_list() == [False] * len(df)


def test_cleanup_only_is_valid() -> None:
    """Cleanup-only filters should still emit an overall validity column in diagnose mode."""
    df = pl.DataFrame({
        "scene": [1, 1, 1],
        "id": [1, 2, 3],
        "frame": [0, 0, 0],
        "category": [AgentCategory.CAR, AgentCategory.PEDESTRIAN, AgentCategory.UNIMPORTANT],
    })

    scene_filter = Filter.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=[AgentCategory.UNIMPORTANT])]
    )
    result = filter_scene(
        df, scene_filter, group_by="scene", category_column="category", mode="diagnose"
    )

    assert len(result) == 2
    assert result["_filter_scene_is_valid"].to_list() == [True, True]


def test_prune_before_validation() -> None:
    """Pruning must happen before validation for the cleaned scene to survive."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 0, 1, 2],
    })

    without_cleanup = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=1)],
        agent_rules=[agent.MaxMissingFrames(maximum=0)],
    )
    with_cleanup = Filter.define(
        cleanup_rules=[cleanup.PruneByRule(rule=agent.MaxMissingFrames(maximum=0))],
        scene_rules=[scene.MinimumAgents(minimum=1)],
        agent_rules=[agent.MaxMissingFrames(maximum=0)],
    )

    assert len(filter_scene(df, without_cleanup, group_by="scene")) == 0

    result = filter_scene(df, with_cleanup, group_by="scene")

    assert result["id"].unique().to_list() == [1]
    assert_frame_equal(result, df.filter(pl.col("id") == 1))


def test_prune_respects_selector() -> None:
    """Scoped pruning should leave out-of-scope agents untouched."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1],
        "id": [1, 1, 2, 2, 3, 3],
        "frame": [0, 1, 0, 1, 0, 1],
        "category": [
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.PEDESTRIAN,
            AgentCategory.PEDESTRIAN,
        ],
    })

    scene_filter = Filter.define(
        cleanup_rules=[
            cleanup.PruneByRule(
                rule=agent.RequireFrames.define(
                    frames=[0, 1, 2], selector=AgentSelector.include(["CAR"])
                )
            )
        ]
    )

    result = filter_scene(df, scene_filter, group_by="scene", category_column="category")

    assert result["id"].unique().to_list() == [3]
    assert result["category"].unique().to_list() == [AgentCategory.PEDESTRIAN]


def test_diagnose_mode_invalid_agents() -> None:
    """Diagnostics should expose which agent rule invalidated the scene."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 0, 1, 2],
    })

    scene_filter = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=1)],
        agent_rules=[agent.MaxMissingFrames(maximum=0)],
    )
    result = filter_scene(df, scene_filter, group_by="scene", mode="diagnose")

    assert result["_filter_rule_agent_max_missing_frames_invalid_agents"].to_list() == [1] * len(df)
    assert result["_filter_rule_agent_max_missing_frames_invalid_fraction"].to_list() == [
        0.5
    ] * len(df)
    assert result["_filter_rule_agent_max_missing_frames_scene_passes"].to_list() == [False] * len(
        df
    )
    assert result["_filter_scene_is_valid"].to_list() == [False] * len(df)
