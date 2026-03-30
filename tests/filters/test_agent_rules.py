import polars as pl
from polars.testing import assert_frame_equal

from dronalize.core.categories import AgentCategory
from dronalize.processing.filters import AgentSelector, Filter, agent, filter_scene, scene, tol


def test_tolerance_keeps_invalid_agents() -> None:
    """Tolerance should keep the full cleaned scene, including invalid agents."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 0, 1, 2],
    })

    scene_filter = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=1)],
        agent_rules=[agent.MaxMissingFrames(maximum=0, tolerance=tol(absolute=1, relative=0.5))],
    )

    result = filter_scene(df, scene_filter, group_by="scene")

    assert len(result) == len(df)
    assert sorted(result["id"].unique().to_list()) == [1, 2]


def test_agent_frames_drop_scene() -> None:
    """Agent frame requirements should invalidate the whole scene."""
    df = pl.DataFrame({"scene": [1, 1, 1, 1], "id": [1, 1, 1, 2], "frame": [10, 12, 14, 10]})

    scene_filter = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)],
        agent_rules=[agent.RequireFrames.define(frames=[0, 4])],
    )
    result = filter_scene(df, scene_filter, group_by="scene")

    assert len(result) == 0


def test_rule_ignores_out_of_scope() -> None:
    """Only scoped agents should contribute to invalid-agent counts."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 0, 1, 2],
        "category": [
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.CAR,
            AgentCategory.PEDESTRIAN,
            AgentCategory.PEDESTRIAN,
            AgentCategory.PEDESTRIAN,
        ],
    })

    scene_filter = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=1)],
        agent_rules=[agent.MaxMissingFrames(maximum=0, selector=AgentSelector.include(["CAR"]))],
    )
    result = filter_scene(df, scene_filter, group_by="scene", category_column="category")

    assert len(result) == len(df)


def test_agent_rule_passes_empty_scope() -> None:
    """A scoped rule should pass when no retained agents are in scope."""
    df = pl.DataFrame({
        "scene": [1, 1, 1],
        "id": [1, 1, 1],
        "frame": [0, 1, 2],
        "category": [AgentCategory.CAR, AgentCategory.CAR, AgentCategory.CAR],
    })

    scene_filter = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=1)],
        agent_rules=[
            agent.MaxMissingFrames(maximum=0, selector=AgentSelector.include(["PEDESTRIAN"]))
        ],
    )
    result = filter_scene(df, scene_filter, group_by="scene", category_column="category")

    assert_frame_equal(result, df)


def test_tolerance_allows_one_invalid() -> None:
    """A crowded scene may survive one bad agent when tolerance is configured."""
    rows: list[dict[str, int]] = []
    for agent_id in range(30):
        frames = [0, 1, 2] if agent_id != 29 else [0, 1]
        rows.extend({"scene": 1, "id": agent_id, "frame": frame} for frame in frames)

    df = pl.DataFrame(rows)

    strict = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)],
        agent_rules=[agent.MaxMissingFrames(maximum=0)],
    )
    assert len(filter_scene(df, strict, group_by="scene")) == 0

    tolerant = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)],
        agent_rules=[agent.MaxMissingFrames(maximum=0, tolerance=tol(absolute=1))],
    )
    tolerant_result = filter_scene(df, tolerant, group_by="scene")
    assert len(tolerant_result) == len(df)


def test_max_gap() -> None:
    """Max-gap checks should fail once an agent has a larger internal gap."""
    df = pl.DataFrame({
        "scene": [1] * 7,
        "id": [1, 1, 1, 2, 2, 2, 2],
        "frame": [0, 2, 3, 0, 1, 2, 3],
    })

    strict = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)], agent_rules=[agent.MaxGap(maximum=0)]
    )
    tolerant = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)], agent_rules=[agent.MaxGap(maximum=1)]
    )

    assert len(filter_scene(df, strict, group_by="scene")) == 0
    assert_frame_equal(filter_scene(df, tolerant, group_by="scene"), df)


def test_min_consecutive_frames() -> None:
    """Longest consecutive run, not total samples, should drive the rule."""
    df = pl.DataFrame({
        "scene": [1] * 9,
        "id": [1, 1, 1, 1, 1, 2, 2, 2, 2],
        "frame": [0, 1, 2, 5, 6, 0, 1, 2, 3],
    })

    passing = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)],
        agent_rules=[agent.MinConsecutiveFrames(minimum=3)],
    )
    failing = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)],
        agent_rules=[agent.MinConsecutiveFrames(minimum=4)],
    )

    assert_frame_equal(filter_scene(df, passing, group_by="scene"), df)
    assert len(filter_scene(df, failing, group_by="scene")) == 0


def test_agent_window_uses_tolerance() -> None:
    """Agent-window coverage should integrate with invalid-agent tolerances."""
    df = pl.DataFrame({
        "scene": [1] * 8,
        "id": [1, 1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 4, 0, 2, 4],
        "category": [AgentCategory.CAR] * 8,
    })

    strict = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)],
        agent_rules=[
            agent.RequireWindow(
                start_frame=0,
                end_frame=4,
                min_fraction=0.8,
                selector=AgentSelector.include(["CAR"]),
            )
        ],
    )
    tolerant = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)],
        agent_rules=[
            agent.RequireWindow(
                start_frame=0,
                end_frame=4,
                min_fraction=0.8,
                selector=AgentSelector.include(["CAR"]),
                tolerance=tol(absolute=1, relative=0.5),
            )
        ],
    )

    assert len(filter_scene(df, strict, group_by="scene", category_column="category")) == 0
    assert_frame_equal(filter_scene(df, tolerant, group_by="scene", category_column="category"), df)


def test_min_samples_drops_scene() -> None:
    """Sample-count checks should fail the scene rather than removing sparse agents."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 2, 2, 2],
        "id": [1, 1, 1, 2, 2, 3, 3, 3],
        "frame": [0, 1, 2, 0, 1, 0, 1, 2],
    })

    scene_filter = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)], agent_rules=[agent.MinSamples(minimum=3)]
    )
    result = filter_scene(df, scene_filter, group_by="scene")

    assert result["scene"].unique().to_list() == [2]
    assert result["id"].unique().to_list() == [3]


def test_starts_by_frame() -> None:
    """Agents that start too late should invalidate the scene."""
    df = pl.DataFrame({"scene": [1] * 5, "id": [1, 1, 1, 2, 2], "frame": [0, 1, 2, 1, 2]})

    strict = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)], agent_rules=[agent.StartsByFrame(frame=0)]
    )
    tolerant = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)], agent_rules=[agent.StartsByFrame(frame=1)]
    )

    assert len(filter_scene(df, strict, group_by="scene")) == 0
    assert_frame_equal(filter_scene(df, tolerant, group_by="scene"), df)


def test_ends_after_frame() -> None:
    """Agents that end too early should invalidate the scene."""
    df = pl.DataFrame({"scene": [1] * 5, "id": [1, 1, 1, 2, 2], "frame": [0, 1, 2, 0, 1]})

    strict = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)], agent_rules=[agent.EndsAfterFrame(frame=2)]
    )
    tolerant = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)], agent_rules=[agent.EndsAfterFrame(frame=1)]
    )

    assert len(filter_scene(df, strict, group_by="scene")) == 0
    assert_frame_equal(filter_scene(df, tolerant, group_by="scene"), df)


def test_min_span_uses_extent() -> None:
    """Span checks should be based on frame extent, not the number of samples."""
    df = pl.DataFrame({"scene": [1] * 5, "id": [1, 1, 1, 2, 2], "frame": [0, 1, 2, 0, 2]})

    passing = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)], agent_rules=[agent.MinSpan(minimum=3)]
    )
    failing = Filter.define(
        scene_rules=[scene.MinimumAgents(minimum=0)], agent_rules=[agent.MinSpan(minimum=4)]
    )

    assert_frame_equal(filter_scene(df, passing, group_by="scene"), df)
    assert len(filter_scene(df, failing, group_by="scene")) == 0
