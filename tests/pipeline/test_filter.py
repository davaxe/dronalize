import polars as pl
from polars.testing import assert_frame_equal

from dronalize.core.categories import AgentCategory
from dronalize.processing.filters import (
    ExcludeAgentCategories,
    Filter,
    MinimumAgents,
    MinimumAgentSamples,
    RequireAgentCoverageAtFrames,
    RequireCompleteAgentCoverage,
    RequireSceneCoverageAtFrames,
    filter_scene,
)
from dronalize.processing.ingest import LoaderConfig


def test_no_scene_filtering_returns_input_for_both_modes() -> None:
    """Filtering disabled should return the original frame unchanged."""
    df = pl.DataFrame({"id": [1, 1], "frame": [0, 1], "scene": [1, 1]})
    config = LoaderConfig(input_len=1, output_len=1, sample_time=0.1, filter=None)

    filtered = filter_scene(df, config.filter, group_by="scene")
    diagnosed = filter_scene(df, config.filter, group_by="scene", mode="diagnose")

    assert_frame_equal(filtered, df)
    assert_frame_equal(diagnosed, df)


def test_cleanup_exclude_agent_categories_removes_only_cleanup_rows() -> None:
    """Cleanup rules should remove explicit categories while preserving the cleaned scene."""
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

    scene_filter = Filter.define(
        cleanup_rules=[ExcludeAgentCategories.define(categories=[AgentCategory.UNIMPORTANT])],
        scene_validation_rules=[MinimumAgents(minimum=2)],
    )

    result = filter_scene(df, scene_filter, group_by="scene", category_column="category")

    assert len(result) == 2
    assert AgentCategory.UNIMPORTANT not in result["category"].to_list()
    assert sorted(result["id"].unique().to_list()) == [1, 2]


def test_min_agents_threshold_applies_after_cleanup() -> None:
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
        cleanup_rules=[ExcludeAgentCategories.define(categories=[AgentCategory.UNIMPORTANT])],
        scene_validation_rules=[MinimumAgents(minimum=2)],
    )

    result = filter_scene(df, scene_filter, group_by="scene", category_column="category")

    assert result["scene"].unique().to_list() == [1]
    assert sorted(result["id"].unique().to_list()) == [1, 2]


def test_filtered_mode_drops_invalid_scenes_and_strips_diagnostics() -> None:
    """Filtered mode should drop failing scenes and remove diagnostic columns."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 2, 2, 2, 2],
        "id": [1, 1, 1, 2, 3, 3, 3, 3],
        "frame": [0, 1, 2, 0, 0, 1, 2, 3],
    })

    scene_filter = Filter.define(
        scene_validation_rules=[MinimumAgents(minimum=1)],
        agent_validation_rules=[RequireCompleteAgentCoverage()],
    )
    result = filter_scene(df, scene_filter, group_by="scene", mode="filtered")

    assert result["scene"].unique().to_list() == [2]
    assert result["id"].unique().to_list() == [3]
    assert all(not column.startswith("_filter_") for column in result.columns)


def test_diagnose_mode_keeps_cleaned_rows_and_adds_validity_column() -> None:
    """Diagnose mode should keep all cleaned rows while annotating scene validity."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 2, 2, 2, 2],
        "id": [1, 1, 1, 2, 3, 3, 3, 3],
        "frame": [0, 1, 2, 0, 0, 1, 2, 3],
    })

    scene_filter = Filter.define(
        scene_validation_rules=[MinimumAgents(minimum=1)],
        agent_validation_rules=[RequireCompleteAgentCoverage()],
    )
    result = filter_scene(df, scene_filter, group_by="scene", mode="diagnose")

    assert result.shape[0] == df.shape[0]
    assert result["_filter_scene_is_valid"].to_list() == ([False] * 4) + ([True] * 4)


def test_diagnose_mode_exposes_scene_rule_columns() -> None:
    """Scene-level rules should contribute their scene-pass diagnostics."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1],
        "id": [1, 1, 1, 2],
        "frame": [10, 12, 14, 10],
    })

    scene_filter = Filter.define(
        scene_validation_rules=[
            MinimumAgents(minimum=0),
            RequireSceneCoverageAtFrames.define(frames=[0, 4]),
        ]
    )
    result = filter_scene(df, scene_filter, group_by="scene", mode="diagnose")

    assert result["_filter_rule_min_agents_scene_passes"].to_list() == [True] * len(df)
    assert result["_filter_rule_scene_frames_scene_passes"].to_list() == [True] * len(df)
    assert result["_filter_scene_is_valid"].to_list() == [True] * len(df)


def test_empty_scene_and_agent_validation_rules_still_produce_valid_diagnostics_after_cleanup() -> (
    None
):
    """Cleanup-only filters should still emit an overall validity column in diagnose mode."""
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

    scene_filter = Filter.define(
        cleanup_rules=[ExcludeAgentCategories.define(categories=[AgentCategory.UNIMPORTANT])]
    )
    result = filter_scene(
        df,
        scene_filter,
        group_by="scene",
        category_column="category",
        mode="diagnose",
    )

    assert len(result) == 2
    assert result["_filter_scene_is_valid"].to_list() == [True, True]


def test_tolerance_keeps_scene_without_pruning_invalid_agents() -> None:
    """Tolerance should keep the full cleaned scene, including invalid agents."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 0, 1, 2],
    })

    scene_filter = Filter.define(
        scene_validation_rules=[MinimumAgents(minimum=1)],
        agent_validation_rules=[
            RequireCompleteAgentCoverage(max_invalid_agents=1, max_invalid_fraction=0.5),
        ],
    )

    result = filter_scene(df, scene_filter, group_by="scene")

    assert len(result) == len(df)
    assert sorted(result["id"].unique().to_list()) == [1, 2]


def test_required_scene_frames_do_not_require_every_agent_to_cover_them() -> None:
    """Scene frame requirements should only validate scene coverage."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1],
        "id": [1, 1, 1, 2],
        "frame": [10, 12, 14, 10],
    })

    scene_filter = Filter.define(
        scene_validation_rules=[
            MinimumAgents(minimum=0),
            RequireSceneCoverageAtFrames.define(frames=[0, 4]),
        ]
    )
    result = filter_scene(df, scene_filter, group_by="scene")

    assert_frame_equal(result, df)


def test_required_agent_frames_drop_scene_when_any_retained_agent_misses_them() -> None:
    """Agent frame requirements should invalidate the whole scene when any agent misses them."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1],
        "id": [1, 1, 1, 2],
        "frame": [10, 12, 14, 10],
    })

    scene_filter = Filter.define(
        scene_validation_rules=[MinimumAgents(minimum=0)],
        agent_validation_rules=[
            RequireAgentCoverageAtFrames.define(frames=[0, 4]),
        ],
    )
    result = filter_scene(df, scene_filter, group_by="scene")

    assert len(result) == 0


def test_crowded_scene_tolerance_allows_single_invalid_agent() -> None:
    """A crowded scene may survive one bad agent when tolerance is configured."""
    rows: list[dict[str, int]] = []
    for agent in range(30):
        frames = [0, 1, 2] if agent != 29 else [0, 1]
        rows.extend({"scene": 1, "id": agent, "frame": frame} for frame in frames)

    df = pl.DataFrame(rows)

    strict = Filter.define(
        scene_validation_rules=[MinimumAgents(minimum=0)],
        agent_validation_rules=[RequireCompleteAgentCoverage()],
    )
    assert len(filter_scene(df, strict, group_by="scene")) == 0

    tolerant = Filter.define(
        scene_validation_rules=[MinimumAgents(minimum=0)],
        agent_validation_rules=[
            RequireCompleteAgentCoverage(max_invalid_agents=1, max_invalid_fraction=0.04),
        ],
    )
    tolerant_result = filter_scene(df, tolerant, group_by="scene")
    assert len(tolerant_result) == len(df)


def test_diagnose_mode_reports_rule_scoped_invalid_agent_counts_and_fraction() -> None:
    """Diagnostics should expose which agent rule invalidated the scene."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 0, 1, 2],
    })

    scene_filter = Filter.define(
        scene_validation_rules=[MinimumAgents(minimum=1)],
        agent_validation_rules=[RequireCompleteAgentCoverage()],
    )
    result = filter_scene(df, scene_filter, group_by="scene", mode="diagnose")

    assert result["_filter_rule_complete_agent_coverage_invalid_agents"].to_list() == [1] * len(df)
    assert result["_filter_rule_complete_agent_coverage_invalid_fraction"].to_list() == [0.5] * len(
        df
    )
    assert result["_filter_rule_complete_agent_coverage_scene_passes"].to_list() == [False] * len(
        df
    )
    assert result["_filter_scene_is_valid"].to_list() == [False] * len(df)


def test_min_samples_per_agent_invalidates_scene_instead_of_pruning_agents() -> None:
    """Sample-count checks should fail the scene rather than removing sparse agents."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 2, 2, 2],
        "id": [1, 1, 1, 2, 2, 3, 3, 3],
        "frame": [0, 1, 2, 0, 1, 0, 1, 2],
    })

    scene_filter = Filter.define(
        scene_validation_rules=[MinimumAgents(minimum=0)],
        agent_validation_rules=[MinimumAgentSamples(minimum=3)],
    )
    result = filter_scene(df, scene_filter, group_by="scene")

    assert result["scene"].unique().to_list() == [2]
    assert result["id"].unique().to_list() == [3]


def test_min_samples_per_agent_via_loader_config() -> None:
    """LoaderConfig should retain the supplied validation rules."""
    config = LoaderConfig(
        input_len=1,
        output_len=1,
        sample_time=0.1,
    ).with_filter(Filter.define(agent_validation_rules=[MinimumAgentSamples(minimum=3)]))

    assert config.filter is not None
    assert config.filter.agent_validation_rules == (MinimumAgentSamples(minimum=3),)
