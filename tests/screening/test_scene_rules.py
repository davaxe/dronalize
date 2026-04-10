from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from dronalize.config.sections import Range
from dronalize.core.categories import AgentCategory
from dronalize.processing.screening import AgentSelector, Screen, scene, screen_scene


def test_keep_frames() -> None:
    """Keep scene when required frames exist."""
    df = pl.DataFrame({"scene": [1, 1, 1, 1], "id": [1, 1, 1, 2], "frame": [10, 12, 14, 10]})

    scene_screening = Screen.define(
        scene_rules=[scene.AgentRange(minimum=0), scene.RequireFrames.define(frames=[0, 4])]
    )
    result = screen_scene(df, scene_screening, group_by="scene")

    assert_frame_equal(result, df)


def test_scope_agents() -> None:
    """Count only selected agents."""
    df = pl.DataFrame({
        "scene": [1, 1],
        "id": [1, 2],
        "frame": [0, 0],
        "category": [AgentCategory.CAR, AgentCategory.PEDESTRIAN],
    })

    scene_screening = Screen.define(
        scene_rules=[scene.AgentRange(minimum=2, selector=AgentSelector.include(["CAR"]))]
    )
    result = screen_scene(df, scene_screening, group_by="scene", category_column="category")

    assert len(result) == 0


def test_window_fraction() -> None:
    """Pass scene when window coverage meets threshold."""
    df = pl.DataFrame({"scene": [1, 1, 1], "id": [1, 1, 1], "frame": [10, 12, 14]})

    passing = Screen.define(
        scene_rules=[
            scene.AgentRange(minimum=0),
            scene.RequireWindow(start_frame=0, end_frame=4, min_fraction=0.6),
        ]
    )
    failing = Screen.define(
        scene_rules=[
            scene.AgentRange(minimum=0),
            scene.RequireWindow(start_frame=0, end_frame=4, min_fraction=0.8),
        ]
    )

    assert_frame_equal(screen_scene(df, passing, group_by="scene"), df)
    assert len(screen_scene(df, failing, group_by="scene")) == 0


def test_allow_gaps() -> None:
    """Allow scene when gap count is within limit."""
    df = pl.DataFrame({"scene": [1, 1, 1], "id": [1, 1, 1], "frame": [0, 2, 3]})

    strict = Screen.define(scene_rules=[scene.MaxMissingFrames()])
    tolerant = Screen.define(scene_rules=[scene.MaxMissingFrames(max_missing_frames=1)])

    assert len(screen_scene(df, strict, group_by="scene")) == 0
    assert_frame_equal(screen_scene(df, tolerant, group_by="scene"), df)


def test_agent_bounds() -> None:
    """Keep scene within agent count bounds."""
    df = pl.DataFrame({"scene": [1, 1, 2, 2, 2], "id": [1, 2, 3, 4, 5], "frame": [0, 0, 0, 0, 0]})

    scene_screening = Screen.define(scene_rules=[scene.AgentRange(minimum=2, maximum=2)])
    result = screen_scene(df, scene_screening, group_by="scene")

    assert result["scene"].unique().to_list() == [1]
    assert sorted(result["id"].unique().to_list()) == [1, 2]


def test_category_bounds() -> None:
    """Enforce category-specific agent ranges."""
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

    scene_screening = Screen.define(
        scene_rules=[
            scene.CategoryRange.define(
                (AgentCategory.CAR, Range(minimum=1, maximum=1)),
                (AgentCategory.PEDESTRIAN, Range(minimum=1)),
            )
        ]
    )
    result = screen_scene(df, scene_screening, group_by="scene", category_column="category")

    assert result["scene"].unique().to_list() == [1]
    assert sorted(result["id"].unique().to_list()) == [1, 2]
