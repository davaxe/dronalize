import polars as pl
from polars.testing import assert_frame_equal

from dronalize.core.categories import AgentCategory
from dronalize.core.models import Range
from dronalize.processing.filtering import AgentSelector, Filter, filter_scene, scene


def test_scene_frames_use_coverage() -> None:
    """Scene frame requirements should only validate scene coverage."""
    df = pl.DataFrame({"scene": [1, 1, 1, 1], "id": [1, 1, 1, 2], "frame": [10, 12, 14, 10]})

    scene_filter = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0), scene.RequireFrames.define(frames=[0, 4])]
    )
    result = filter_scene(df, scene_filter, group_by="scene")

    assert_frame_equal(result, df)


def test_min_agents_respects_selector() -> None:
    """Category-scoped minimum-agent checks should only count matching agents."""
    df = pl.DataFrame({
        "scene": [1, 1],
        "id": [1, 2],
        "frame": [0, 0],
        "category": [AgentCategory.CAR, AgentCategory.PEDESTRIAN],
    })

    scene_filter = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=2, selector=AgentSelector.include(["CAR"]))]
    )
    result = filter_scene(df, scene_filter, group_by="scene", category_column="category")

    assert len(result) == 0


def test_scene_window_uses_fraction() -> None:
    """Scene-window coverage should pass once the minimum fraction is met."""
    df = pl.DataFrame({"scene": [1, 1, 1], "id": [1, 1, 1], "frame": [10, 12, 14]})

    passing = Filter.define(
        scene_rules=[
            scene.MinimumAgents(minimum=0),
            scene.RequireWindow(start_frame=0, end_frame=4, min_fraction=0.6),
        ]
    )
    failing = Filter.define(
        scene_rules=[
            scene.MinimumAgents(minimum=0),
            scene.RequireWindow(start_frame=0, end_frame=4, min_fraction=0.8),
        ]
    )

    assert_frame_equal(filter_scene(df, passing, group_by="scene"), df)
    assert len(filter_scene(df, failing, group_by="scene")) == 0


def test_scene_gap_tolerance() -> None:
    """Gap tolerance should allow a bounded number of missing scene frames."""
    df = pl.DataFrame({"scene": [1, 1, 1], "id": [1, 1, 1], "frame": [0, 2, 3]})

    strict = Filter.define(scene_rules=[scene.MaxMissingFrames()])
    tolerant = Filter.define(scene_rules=[scene.MaxMissingFrames(max_missing_frames=1)])

    assert len(filter_scene(df, strict, group_by="scene")) == 0
    assert_frame_equal(filter_scene(df, tolerant, group_by="scene"), df)


def test_agent_range_bounds() -> None:
    """Agent-range checks should keep only scenes within the configured bounds."""
    df = pl.DataFrame({"scene": [1, 1, 2, 2, 2], "id": [1, 2, 3, 4, 5], "frame": [0, 0, 0, 0, 0]})

    scene_filter = Filter.define(scene_rules=[scene.AgentRange(minimum=2, maximum=2)])
    result = filter_scene(df, scene_filter, group_by="scene")

    assert result["scene"].unique().to_list() == [1]
    assert sorted(result["id"].unique().to_list()) == [1, 2]


def test_category_range_per_category() -> None:
    """Category-range checks should enforce per-category agent counts."""
    df = pl.DataFrame({
        "scene": [1, 1, 2, 2],
        "id": [1, 2, 3, 4],
        "frame": [0, 0, 0, 0],
        "category": [
            AgentCategory.CAR,
            AgentCategory.PEDESTRIAN,
            AgentCategory.CAR,
            AgentCategory.CAR,
        ],
    })

    scene_filter = Filter.define(
        scene_rules=[
            scene.CategoryRange(
                ranges={"CAR": Range(minimum=1, maximum=1), "PEDESTRIAN": Range(minimum=1)}
            )
        ]
    )
    result = filter_scene(df, scene_filter, group_by="scene", category_column="category")

    assert result["scene"].unique().to_list() == [1]
    assert sorted(result["id"].unique().to_list()) == [1, 2]
