from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dronalize.config.models import (
    LaneChangeConfig,
    ScenesConfig,
    SplitWeights,
    TimeBlockAssign,
    WindowConfig,
)
from dronalize.processing.models import AssignmentRequest, PipelinePlan
from dronalize.processing.pipeline.trajectory import build_trajectory_pipeline

if TYPE_CHECKING:
    import polars as pl

    from tests.support import DataFramePresets

SceneDict = dict[str, list[Any]]


def _scenes(
    *,
    history_frames: int = 2,
    future_frames: int = 1,
    window_step: int | None = 1,
    lane_change: LaneChangeConfig | None = None,
) -> ScenesConfig:
    return ScenesConfig(
        history_frames=history_frames,
        future_frames=future_frames,
        sample_time=1.0,
        window=WindowConfig(step=window_step) if window_step is not None else None,
        lane_change=lane_change,
    )


def _run_pipeline(
    frame: pl.DataFrame,
    *,
    plan: PipelinePlan,
    window_by: str | None = None,
) -> list[SceneDict]:
    return [
        scene.to_dict(as_series=False)
        for scene in build_trajectory_pipeline(plan, window_by=window_by).execute(
            frame, collect=True
        )
    ]


def test_standard_trajectory_pipeline_outputs_windowed_scenes(
    scene_df_presets: DataFramePresets,
) -> None:
    frame: pl.DataFrame = scene_df_presets["single_agent_windowed"]()
    scenes = _run_pipeline(frame, plan=PipelinePlan(scenes=_scenes()))

    assert scenes == [
        {
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "agent_category": [4, 4, 4],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
        },
        {
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "agent_category": [4, 4, 4],
            "x": [1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0],
        },
    ]


def test_standard_trajectory_pipeline_outputs_split_labeled_scenes(
    scene_df_presets: DataFramePresets,
) -> None:
    frame: pl.DataFrame = scene_df_presets["single_agent_time_split"]()
    scenes = _run_pipeline(
        frame,
        plan=PipelinePlan(
            scenes=_scenes(),
            assignment=AssignmentRequest(
                config=TimeBlockAssign(ratio=SplitWeights(train=0.5, val=0.5))
            ),
        ),
    )

    assert scenes == [
        {
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "agent_category": [4, 4, 4],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "split": ["train", "train", "train"],
        },
        {
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "agent_category": [4, 4, 4],
            "x": [1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0],
            "split": ["train", "train", "train"],
        },
        {
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "agent_category": [4, 4, 4],
            "x": [4.0, 5.0, 6.0],
            "y": [0.0, 0.0, 0.0],
            "split": ["val", "val", "val"],
        },
        {
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "agent_category": [4, 4, 4],
            "x": [5.0, 6.0, 7.0],
            "y": [0.0, 0.0, 0.0],
            "split": ["val", "val", "val"],
        },
    ]


def test_lane_change_sampling_keeps_lane_change_windows_and_thins_steady_windows(
    scene_df_presets: DataFramePresets,
) -> None:
    frame = scene_df_presets["lane_change_sequences"]()
    plan = PipelinePlan(
        scenes=_scenes(
            lane_change=LaneChangeConfig(persist=1, negative_keep_every=2, required_lane_changes=1)
        )
    )

    scenes = _run_pipeline(frame, plan=plan, window_by="sequence")

    steady_scenes = [scene for scene in scenes if scene["sequence"][0] == "steady"]
    changing_scenes = [scene for scene in scenes if scene["sequence"][0] == "changing"]

    assert len(changing_scenes) == 3
    assert len(steady_scenes) == 2

    assert scenes == [
        {
            "sequence": ["steady", "steady", "steady"],
            "frame": [0, 1, 2],
            "id": [2, 2, 2],
            "agent_category": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "lane_id": [1, 1, 1],
        },
        {
            "sequence": ["steady", "steady", "steady"],
            "frame": [0, 1, 2],
            "id": [2, 2, 2],
            "agent_category": [1, 1, 1],
            "x": [2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 0.0],
            "lane_id": [1, 1, 1],
        },
        {
            "sequence": ["changing", "changing", "changing"],
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "agent_category": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "lane_id": [1, 1, 2],
        },
        {
            "sequence": ["changing", "changing", "changing"],
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "agent_category": [1, 1, 1],
            "x": [1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0],
            "lane_id": [1, 2, 2],
        },
        {
            "sequence": ["changing", "changing", "changing"],
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "agent_category": [1, 1, 1],
            "x": [2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 0.0],
            "lane_id": [2, 2, 2],
        },
    ]


def test_lane_change_sampling_requires_window_sampling(
    scene_df_presets: DataFramePresets,
) -> None:
    frame = scene_df_presets["lane_change_sequences"]()
    plan = PipelinePlan(
        scenes=_scenes(
            window_step=None,
            lane_change=LaneChangeConfig(persist=1, negative_keep_every=2),
        )
    )

    with pytest.raises(ValueError, match="requires window sampling"):
        _ = _run_pipeline(frame, plan=plan, window_by="sequence")


def test_lane_change_keep_every_one_matches_standard_pipeline(
    scene_df_presets: DataFramePresets,
) -> None:
    frame = scene_df_presets["lane_change_sequences"]()
    standard_plan = PipelinePlan(scenes=_scenes())
    lane_change_plan = PipelinePlan(
        scenes=_scenes(lane_change=LaneChangeConfig(persist=1, negative_keep_every=1))
    )

    assert _run_pipeline(frame, plan=lane_change_plan, window_by="sequence") == _run_pipeline(
        frame, plan=standard_plan, window_by="sequence"
    )
