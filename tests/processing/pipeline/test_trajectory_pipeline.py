from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dronalize.config.models import (
    LaneChangeConfig,
    ScenesConfig,
    SplitWeights,
    TimeSplitConfig,
    WindowConfig,
)
from dronalize.processing.models import PipelinePlan, SplitRequest
from dronalize.processing.pipeline.spec import lane_change_sampling, standard, trajectory_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from tests.support import DataFramePresets

SceneDict = dict[str, list[Any]]


def _scenes(
    *,
    history_frames: int = 2,
    future_frames: int = 1,
    window_step: int = 1,
    lane_change: LaneChangeConfig | None = None,
) -> ScenesConfig:
    return ScenesConfig(
        history_frames=history_frames,
        future_frames=future_frames,
        sample_time=1.0,
        window=WindowConfig(step=window_step),
        lane_change=lane_change,
    )


def _run_pipeline(
    frame: pl.DataFrame,
    *,
    plan: PipelinePlan,
    spec_builder: Callable[..., Any] = standard,
    window_by: str | None = None,
) -> list[SceneDict]:
    spec = spec_builder(plan, window_by=window_by)
    return [
        scene.to_dict(as_series=False)
        for scene in trajectory_pipeline(spec).execute(frame, collect=True)
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
            split=SplitRequest(config=TimeSplitConfig(ratio=SplitWeights(train=0.5, val=0.5))),
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
            lane_change=LaneChangeConfig(
                persist=1,
                negative_keep_every=2,
                required_lane_changes=1,
            )
        )
    )

    scenes = _run_pipeline(
        frame,
        plan=plan,
        spec_builder=lane_change_sampling,
        window_by="sequence",
    )

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
