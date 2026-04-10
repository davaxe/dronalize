import polars as pl
from polars.testing import assert_frame_equal

from dronalize.core.categories import AgentCategory
from dronalize.processing.screening import AgentSelector, Screen, agent, cleanup, screen_scene


def test_exclude_categories_removes_matching_rows() -> None:
    """Exclude cleanup should drop only rows in the listed categories."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1],
        "id": [1, 2, 3, 4],
        "frame": [0, 0, 0, 0],
        "category": [
            AgentCategory.CAR,
            AgentCategory.PEDESTRIAN,
            AgentCategory.UNIMPORTANT,
            AgentCategory.BUS,
        ],
    })

    scene_screening = Screen.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=[AgentCategory.UNIMPORTANT])]
    )

    result = screen_scene(df, scene_screening, group_by="scene", category_column="category")

    assert_frame_equal(result, df.filter(pl.col("category") != AgentCategory.UNIMPORTANT))


def test_include_categories_keeps_only_requested_rows() -> None:
    """Include cleanup should retain only rows in the listed categories."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1],
        "id": [1, 2, 3, 4],
        "frame": [0, 0, 0, 0],
        "category": [
            AgentCategory.CAR,
            AgentCategory.PEDESTRIAN,
            AgentCategory.CAR,
            AgentCategory.BUS,
        ],
    })

    scene_screening = Screen.define(
        cleanup_rules=[
            cleanup.IncludeCategories.define(
                categories=[AgentCategory.CAR, AgentCategory.PEDESTRIAN]
            )
        ]
    )

    result = screen_scene(df, scene_screening, group_by="scene", category_column="category")

    assert_frame_equal(
        result, df.filter(pl.col("category").is_in([AgentCategory.CAR, AgentCategory.PEDESTRIAN]))
    )


def test_prune_by_rule_removes_all_rows_for_failing_agents() -> None:
    """Pruning should remove the whole agent, not just a subset of its rows."""
    df = pl.DataFrame({"scene": [1, 1, 1, 1, 1], "id": [1, 1, 1, 2, 2], "frame": [0, 1, 2, 0, 2]})

    scene_screening = Screen.define(
        cleanup_rules=[cleanup.PruneByRule(agent_rule=agent.MaxMissingFrames(maximum=0))]
    )

    result = screen_scene(df, scene_screening, group_by="scene")

    assert_frame_equal(result, df.filter(pl.col("id") == 1))


def test_prune_by_rule_respects_selector_scope() -> None:
    """Scoped pruning should ignore agents outside the wrapped rule selector."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1],
        "id": [1, 1, 2, 2, 3, 3],
        "frame": [0, 2, 0, 2, 0, 1],
        "category": [
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.PEDESTRIAN,
            AgentCategory.PEDESTRIAN,
        ],
    })

    scene_screening = Screen.define(
        cleanup_rules=[
            cleanup.PruneByRule(
                agent_rule=agent.RequireFrames.define(
                    frames=[0, 1, 2], selector=AgentSelector.include(["CAR"])
                )
            )
        ]
    )

    result = screen_scene(df, scene_screening, group_by="scene", category_column="category")

    assert_frame_equal(result, df.filter(pl.col("category") == AgentCategory.PEDESTRIAN))


def test_prune_by_rule_evaluates_agents_within_each_scene() -> None:
    """Pruning should be scene-local even when the same agent id appears in multiple scenes."""
    df = pl.DataFrame({"scene": [1, 1, 2, 2, 2], "id": [7, 7, 7, 7, 8], "frame": [0, 2, 0, 1, 0]})

    scene_screening = Screen.define(
        cleanup_rules=[cleanup.PruneByRule(agent_rule=agent.RequireFrames.define(frames=[0, 1]))]
    )

    result = screen_scene(df, scene_screening, group_by="scene")

    assert_frame_equal(result, pl.DataFrame({"scene": [2, 2], "id": [7, 7], "frame": [0, 1]}))
