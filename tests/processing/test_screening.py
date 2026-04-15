from __future__ import annotations

import polars as pl

from dronalize.core import AgentCategory
from dronalize.processing.screening import Screen, agent, cleanup, scene, screen_scene, tol
from dronalize.processing.screening.apply import AGENT_PASS_COLUMN


def test_cleanup_exclude_categories_removes_only_matching_rows() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 1],
        "id": [1, 2, 3],
        "frame": [0, 0, 0],
        "category": [AgentCategory.CAR, AgentCategory.UNIMPORTANT, AgentCategory.BUS],
    })
    rules = Screen.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=[AgentCategory.UNIMPORTANT])]
    )

    screened = screen_scene(df, rules, group_by="scene", category_column="category")

    assert screened["id"].to_list() == [1, 3]


def test_agent_rule_tolerance_keeps_scene_but_marks_failed_agents() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 0, 1, 2],
    })

    strict = Screen.define(
        scene_rules=[scene.AgentRange(minimum=1)],
        agent_rules=[agent.MaxMissingFrames(maximum=0)],
    )
    tolerant = Screen.define(
        scene_rules=[scene.AgentRange(minimum=1)],
        agent_rules=[agent.MaxMissingFrames(maximum=0, tolerance=tol(absolute=1, relative=0.5))],
    )

    strict_result = screen_scene(df, strict, group_by="scene")
    tolerant_result = screen_scene(df, tolerant, group_by="scene", mark_passed_agents=True)

    assert strict_result.is_empty()
    assert len(tolerant_result) == len(df)
    assert AGENT_PASS_COLUMN in tolerant_result.columns
    assert tolerant_result.filter(pl.col("id") == 1)[AGENT_PASS_COLUMN].all()
    assert not tolerant_result.filter(pl.col("id") == 2)[AGENT_PASS_COLUMN].all()


def test_scene_agent_range_filters_out_of_bounds_scenes() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 2, 2, 2],
        "id": [10, 11, 20, 21, 22],
        "frame": [0, 0, 0, 0, 0],
    })

    rules = Screen.define(scene_rules=[scene.AgentRange(minimum=2, maximum=2)])
    screened = screen_scene(df, rules, group_by="scene")

    assert screened["scene"].unique().to_list() == [1]
