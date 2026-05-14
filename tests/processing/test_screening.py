from __future__ import annotations

import polars as pl

from dronalize.config.models import Tolerance
from dronalize.core import AgentCategory
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.screening import ScreeningRuleSet, agent, cleanup, scene, screen_scene
from dronalize.processing.screening.screen import (
    AGENT_SCREENING_PASS_COLUMN,
    SCENE_SCREENING_PASS_COLUMN,
)


def test_cleanup_exclude_categories_removes_only_matching_rows() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 1],
        "id": [1, 2, 3],
        "frame": [0, 0, 0],
        "category": [AgentCategory.CAR, AgentCategory.UNIMPORTANT, AgentCategory.BUS],
    })
    rules = ScreeningRuleSet.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=[AgentCategory.UNIMPORTANT])]
    )

    screened = screen_scene(
        df, rules, scene_group_by="scene", columns=TrajectoryColumns(category="category")
    )

    assert screened["id"].to_list() == [1, 3]


def test_agent_rule_tolerance_keeps_scene_but_marks_failed_agents() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 0, 1, 2],
    })

    strict = ScreeningRuleSet.define(
        scene_rules=[scene.AgentRange(minimum=1)],
        agent_rules=[agent.AgentMaxMissingFrames(maximum=0)],
    )
    tolerant = ScreeningRuleSet.define(
        scene_rules=[scene.AgentRange(minimum=1)],
        agent_rules=[
            agent.AgentMaxMissingFrames(maximum=0, tolerance=Tolerance(absolute=1, relative=0.5))
        ],
    )

    strict_result = screen_scene(df, strict, scene_group_by="scene", columns=TrajectoryColumns())
    tolerant_result = screen_scene(
        df, tolerant, scene_group_by="scene", mark_passed_agents=True, columns=TrajectoryColumns()
    )

    assert strict_result.is_empty()
    assert len(tolerant_result) == len(df)
    assert AGENT_SCREENING_PASS_COLUMN in tolerant_result.columns
    assert tolerant_result.filter(pl.col("id") == 1)[AGENT_SCREENING_PASS_COLUMN].all()
    assert not tolerant_result.filter(pl.col("id") == 2)[AGENT_SCREENING_PASS_COLUMN].all()


def test_scene_agent_range_filters_out_of_bounds_scenes() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 2, 2, 2],
        "id": [10, 11, 20, 21, 22],
        "frame": [0, 0, 0, 0, 0],
    })

    rules = ScreeningRuleSet.define(scene_rules=[scene.AgentRange(minimum=2, maximum=2)])
    screened = screen_scene(df, rules, scene_group_by="scene", columns=TrajectoryColumns())

    assert screened["scene"].unique().to_list() == [1]


def test_screen_scene_can_retain_scene_pass_flags_for_runtime() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 2, 2, 2],
        "id": [10, 11, 20, 21, 22],
        "frame": [0, 0, 0, 0, 0],
    })

    rules = ScreeningRuleSet.define(scene_rules=[scene.AgentRange(minimum=2, maximum=2)])
    screened = screen_scene(
        df, rules, scene_group_by="scene", columns=TrajectoryColumns(), retain_scene_passes=True
    )

    assert SCENE_SCREENING_PASS_COLUMN in screened.columns
    assert screened.group_by("scene").agg(pl.col(SCENE_SCREENING_PASS_COLUMN).first()).sort(
        "scene"
    ).rows() == [(1, True), (2, False)]


def test_agent_min_distance() -> None:
    df = pl.DataFrame({
        "id": [1, 1, 2, 2, 3, 3],
        "frame": [0, 1, 0, 1, 0, 1],
        "x": [0.0, 0.0, 10.0, 10.0, 0, 10.0],
        "y": [0.0, 0.0, 10.0, 10.0, 0, 10.0],
    })

    rules = ScreeningRuleSet.define(cleanup_rules=[agent.MinDistance(minimum=14)])
    screened = screen_scene(df, rules, columns=TrajectoryColumns(x="x", y="y"))
    assert screened["id"].unique().to_list() == [3]

    rules = ScreeningRuleSet.define(
        agent_rules=[agent.MinDistance(minimum=14, tolerance=Tolerance(absolute=2))]
    )
    screened = screen_scene(df, rules, columns=TrajectoryColumns(x="x", y="y"))
    assert screened["id"].unique().to_list() == [1, 2, 3]
