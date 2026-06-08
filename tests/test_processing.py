from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import polars as pl
import pytest

from dronalize.config.models import PassingRequirement, Tolerance
from dronalize.core import AgentCategory
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.loading.assigner import StatelessWeightedAssigner
from dronalize.processing.screening import (
    AgentCategorySelector,
    ScreeningRuleSet,
    agent,
    cleanup,
    scene,
)
from dronalize.processing.screening.screen import screen_data

if TYPE_CHECKING:
    from collections.abc import Sequence


def _screen_scene(
    df: pl.DataFrame,
    rules: ScreeningRuleSet | None,
    *,
    columns: TrajectoryColumns,
    scene_group_by: str | Sequence[str] | None = None,
) -> pl.DataFrame:
    if rules is None:
        return df

    if scene_group_by is None:
        result = screen_data(df, rules, columns=columns)
        return result.frame if result.passes_scene else df.clear()

    group_columns = [scene_group_by] if isinstance(scene_group_by, str) else list(scene_group_by)
    groups = df.partition_by(group_columns, maintain_order=True, as_dict=False)
    kept = [
        result.frame
        for group in groups
        if (result := screen_data(group, rules, columns=columns)).passes_scene
    ]
    return pl.concat(kept) if kept else df.clear()


def test_exclude_categories_removes_matches() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 1],
        "id": [1, 2, 3],
        "frame": [0, 0, 0],
        "category": [AgentCategory.CAR, AgentCategory.UNIMPORTANT, AgentCategory.BUS],
    })
    rules = ScreeningRuleSet.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=[AgentCategory.UNIMPORTANT])]
    )

    screened = _screen_scene(
        df, rules, scene_group_by="scene", columns=TrajectoryColumns(category="category")
    )

    assert screened["id"].to_list() == [1, 3]


def test_cleanup_stats_capture_removed_rows_and_agents() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 0, 1, 2],
        "category": [
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.UNIMPORTANT,
            AgentCategory.UNIMPORTANT,
            AgentCategory.UNIMPORTANT,
        ],
    })
    rules = ScreeningRuleSet.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=[AgentCategory.UNIMPORTANT])]
    )

    result = screen_data(df, rules, columns=TrajectoryColumns(category="category"))
    cleanup_stats = result.cleanup

    assert cleanup_stats is not None
    assert cleanup_stats.rows_before == 6
    assert cleanup_stats.rows_after == 3
    assert cleanup_stats.rows_removed == 3
    assert cleanup_stats.agents_before == 2
    assert cleanup_stats.agents_after == 1
    assert cleanup_stats.agents_removed == 1
    assert len(cleanup_stats.by_rule) == 1
    assert cleanup_stats.by_rule[0].rule_name == "cleanup_exclude"
    assert result.frame["id"].to_list() == [1, 1, 1]


def test_screening_result_carries_cleanup_metadata_without_internal_columns() -> None:
    df = pl.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 0, 1, 2],
        "category": [
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.UNIMPORTANT,
            AgentCategory.UNIMPORTANT,
            AgentCategory.UNIMPORTANT,
        ],
    })
    rules = ScreeningRuleSet.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=[AgentCategory.UNIMPORTANT])]
    )

    result = screen_data(df, rules, columns=TrajectoryColumns(category="category"))

    assert result.passes_scene
    assert result.cleanup is not None
    assert result.cleanup.rows_before == 6
    assert result.cleanup.rows_removed == 3
    assert result.cleanup.agents_before == 2
    assert result.cleanup.agents_removed == 1
    assert not any(column.startswith("_cleanup") for column in result.frame.columns)
    assert not any(column.startswith("_screening") for column in result.frame.columns)


def test_tolerance_marks_failed_agents() -> None:
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

    strict_result = _screen_scene(df, strict, scene_group_by="scene", columns=TrajectoryColumns())
    tolerant_result = _screen_scene(
        df, tolerant, scene_group_by="scene", columns=TrajectoryColumns()
    )
    tolerant_metadata = screen_data(df, tolerant, columns=TrajectoryColumns())

    assert strict_result.is_empty()
    assert len(tolerant_result) == len(df)
    assert tolerant_metadata.passes_scene
    assert tolerant_metadata.passed_agent_ids == frozenset({1})


def test_require_keeps_scene_with_enough_agents() -> None:
    df = pl.DataFrame({"scene": [1, 1, 1, 1, 1], "id": [1, 1, 2, 2, 3], "frame": [0, 1, 0, 1, 0]})
    rules = ScreeningRuleSet.define(
        agent_rules=[agent.MinObservations(minimum=2, require=PassingRequirement(absolute=2))]
    )

    result = _screen_scene(df, rules, scene_group_by="scene", columns=TrajectoryColumns())
    metadata = screen_data(df, rules, columns=TrajectoryColumns())

    assert len(result) == len(df)
    assert metadata.passes_scene
    assert metadata.passed_agent_ids == frozenset({1, 2})


def test_require_frames_marks_agents_using_scene_relative_frames() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 1, 2, 3],
    })
    rules = ScreeningRuleSet.define(
        agent_rules=[
            agent.AgentRequireFrames.define(frames={3}, require=PassingRequirement(absolute=1))
        ]
    )

    result = _screen_scene(df, rules, scene_group_by="scene", columns=TrajectoryColumns())
    metadata = screen_data(df, rules, columns=TrajectoryColumns())

    assert len(result) == len(df)
    assert metadata.passes_scene
    assert metadata.passed_agent_ids == frozenset({2})


def test_require_frames_uses_pre_cleanup_scene_start() -> None:
    df = pl.DataFrame({"scene": [1, 1, 1, 1], "id": [1, 2, 2, 2], "frame": [0, 1, 2, 3]})
    rules = ScreeningRuleSet.define(
        cleanup_rules=[agent.MinObservations(minimum=2)],
        agent_rules=[
            agent.AgentRequireFrames.define(frames={3}, require=PassingRequirement(absolute=1))
        ],
    )

    result = _screen_scene(df, rules, scene_group_by="scene", columns=TrajectoryColumns())
    metadata = screen_data(df, rules, columns=TrajectoryColumns())

    assert result["id"].unique().to_list() == [2]
    assert metadata.passes_scene
    assert metadata.passed_agent_ids == frozenset({2})


def test_require_relative_filters_scene() -> None:
    df = pl.DataFrame({"scene": [1, 1, 1, 1, 1], "id": [1, 1, 2, 2, 3], "frame": [0, 1, 0, 1, 0]})
    rules = ScreeningRuleSet.define(
        agent_rules=[agent.MinObservations(minimum=2, require=PassingRequirement(relative=0.75))]
    )

    result = _screen_scene(df, rules, scene_group_by="scene", columns=TrajectoryColumns())

    assert result.is_empty()


def test_require_and_tolerance_both_apply() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 2, 2, 3, 3, 4, 5],
        "frame": [0, 1, 0, 1, 0, 1, 0, 0],
    })
    rule = agent.MinObservations(
        minimum=2, require=PassingRequirement(absolute=3), tolerance=Tolerance(relative=0.25)
    )

    result = _screen_scene(
        df,
        ScreeningRuleSet.define(agent_rules=[rule]),
        scene_group_by="scene",
        columns=TrajectoryColumns(),
    )

    assert result.is_empty()


def test_require_fails_without_matches() -> None:
    df = pl.DataFrame({
        "scene": [1, 1],
        "id": [1, 1],
        "frame": [0, 1],
        "category": [AgentCategory.CAR, AgentCategory.CAR],
    })
    rules = ScreeningRuleSet.define(
        agent_rules=[
            agent.MinObservations(
                minimum=2,
                selector=AgentCategorySelector.include(AgentCategory.PEDESTRIAN),
                require=PassingRequirement(absolute=1),
            )
        ]
    )

    result = _screen_scene(
        df, rules, scene_group_by="scene", columns=TrajectoryColumns(category="category")
    )

    assert result.is_empty()


def test_prune_by_rejects_agent_require() -> None:
    with pytest.raises(ValueError, match="require"):
        _ = cleanup.PruneByRule(
            agent_rule=agent.MinObservations(minimum=2, require=PassingRequirement(absolute=1))
        )


def test_scene_agent_range_filters_scenes() -> None:
    df = pl.DataFrame({
        "scene": [1, 1, 2, 2, 2],
        "id": [10, 11, 20, 21, 22],
        "frame": [0, 0, 0, 0, 0],
    })

    rules = ScreeningRuleSet.define(scene_rules=[scene.AgentRange(minimum=2, maximum=2)])
    screened = _screen_scene(df, rules, scene_group_by="scene", columns=TrajectoryColumns())

    assert screened["scene"].unique().to_list() == [1]


def test_screening_result_reports_scene_pass() -> None:
    df = pl.DataFrame({"id": [20, 21, 22], "frame": [0, 0, 0]})

    rules = ScreeningRuleSet.define(scene_rules=[scene.AgentRange(minimum=2, maximum=2)])
    result = screen_data(df, rules, columns=TrajectoryColumns())

    assert not result.passes_scene


def test_agent_min_distance() -> None:
    df = pl.DataFrame({
        "id": [1, 1, 2, 2, 3, 3],
        "frame": [0, 1, 0, 1, 0, 1],
        "x": [0.0, 0.0, 10.0, 10.0, 0, 10.0],
        "y": [0.0, 0.0, 10.0, 10.0, 0, 10.0],
    })

    rules = ScreeningRuleSet.define(cleanup_rules=[agent.MinDistance(minimum=14)])
    screened = _screen_scene(df, rules, columns=TrajectoryColumns(x="x", y="y"))
    assert screened["id"].unique().to_list() == [3]

    rules = ScreeningRuleSet.define(
        agent_rules=[agent.MinDistance(minimum=14, tolerance=Tolerance(absolute=2))]
    )
    screened = _screen_scene(df, rules, columns=TrajectoryColumns(x="x", y="y"))
    assert screened["id"].unique().to_list() == [1, 2, 3]


def test_weighted_assigner_is_deterministic() -> None:
    assigner_a = StatelessWeightedAssigner(groups=["A", "B", "C"], weights=[0.5, 0.3, 0.2], seed=42)
    assigner_b = StatelessWeightedAssigner(groups=["A", "B", "C"], weights=[0.5, 0.3, 0.2], seed=42)

    values = list(range(200))
    assert [assigner_a.assign(v) for v in values] == [assigner_b.assign(v) for v in values]


def test_weighted_assigner_matches_weights() -> None:
    assigner = StatelessWeightedAssigner(groups=["A", "B", "C"], weights=[0.5, 0.3, 0.2], seed=100)

    counts = Counter(assigner.assign(v) for v in range(10_000))
    total = sum(counts.values())

    assert math.isclose(counts["A"] / total, 0.5, abs_tol=0.02)
    assert math.isclose(counts["B"] / total, 0.3, abs_tol=0.02)
    assert math.isclose(counts["C"] / total, 0.2, abs_tol=0.02)


def test_weighted_assigner_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _ = StatelessWeightedAssigner(groups=["A", "B"], weights=[1.0, -1.0], seed=0)

    with pytest.raises(ValueError, match="At least one weight"):
        _ = StatelessWeightedAssigner(groups=["A", "B"], weights=[0.0, 0.0], seed=0)
