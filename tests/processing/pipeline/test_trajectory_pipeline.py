from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import polars.selectors as cs
import pytest
from polars.testing import assert_frame_equal

from dronalize.config.models import (
    LaneChangeConfig,
    ScenesConfig,
    SplitWeights,
    TimeBlockAssign,
    WindowConfig,
)
from dronalize.core.functional.basic import normalize_group_by
from dronalize.core.functional.window import sliding_window
from dronalize.processing.models import SplitAssignmentPlan, TrajectoryPipelinePlan
from dronalize.processing.pipeline import transforms as tr
from dronalize.processing.pipeline.trajectory import build_trajectory_pipeline

if TYPE_CHECKING:
    from dronalize.core.functional.window import WindowPolicy
    from tests.support import DataFramePresets

SceneDict = dict[str, list[Any]]


def _scenes(
    *,
    horizon_frames: int = 3,
    default_observation_length: int | None = 2,
    window_step: int | None = 1,
    window_policy: WindowPolicy = "strict",
    lane_change: LaneChangeConfig | None = None,
) -> ScenesConfig:
    return ScenesConfig(
        horizon_frames=horizon_frames,
        default_observation_length=default_observation_length,
        sample_time=1.0,
        window=(
            WindowConfig(step=window_step, policy=window_policy)
            if window_step is not None
            else None
        ),
        lane_change=lane_change,
    )


def _run_pipeline(
    frame: pl.DataFrame, *, plan: TrajectoryPipelinePlan, window_by: str | None = None
) -> list[SceneDict]:
    return [
        scene.to_dict(as_series=False)
        for scene in build_trajectory_pipeline(plan, window_by=window_by).execute(
            frame, collect=True
        )
    ]


def test_pipeline_outputs_windows(scene_df_presets: DataFramePresets) -> None:
    frame: pl.DataFrame = scene_df_presets["single_agent_windowed"]()
    scenes = _run_pipeline(frame, plan=TrajectoryPipelinePlan(scenes=_scenes()))

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


def test_partial_window_keeps_edge_windows() -> None:
    frame = pl.DataFrame({
        "frame": [0, 1, 2],
        "id": [1, 1, 1],
        "agent_category": [4, 4, 4],
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 0.0, 0.0],
    })

    scenes = _run_pipeline(
        frame,
        plan=TrajectoryPipelinePlan(
            scenes=_scenes(
                horizon_frames=5,
                default_observation_length=2,
                window_step=1,
                window_policy="partial",
            )
        ),
    )

    assert scenes == [
        {
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "agent_category": [4, 4, 4],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
        },
        {"frame": [0, 1], "id": [1, 1], "agent_category": [4, 4], "x": [1.0, 2.0], "y": [0.0, 0.0]},
        {"frame": [0], "id": [1], "agent_category": [4], "x": [2.0], "y": [0.0]},
    ]


def test_pipeline_outputs_split_labels(scene_df_presets: DataFramePresets) -> None:
    frame: pl.DataFrame = scene_df_presets["single_agent_time_split"]()
    scenes = _run_pipeline(
        frame,
        plan=TrajectoryPipelinePlan(
            scenes=_scenes(),
            assignment=SplitAssignmentPlan(
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


def test_lane_change_sampling_thins_steady_windows(scene_df_presets: DataFramePresets) -> None:
    frame = scene_df_presets["lane_change_sequences"]()
    plan = TrajectoryPipelinePlan(
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


def test_lane_change_sampling_requires_windows(scene_df_presets: DataFramePresets) -> None:
    frame = scene_df_presets["lane_change_sequences"]()
    plan = TrajectoryPipelinePlan(
        scenes=_scenes(
            window_step=None, lane_change=LaneChangeConfig(persist=1, negative_keep_every=2)
        )
    )

    with pytest.raises(ValueError, match="requires window sampling"):
        _ = _run_pipeline(frame, plan=plan, window_by="sequence")


def test_keep_every_one_matches_standard(scene_df_presets: DataFramePresets) -> None:
    frame = scene_df_presets["lane_change_sequences"]()
    standard_plan = TrajectoryPipelinePlan(scenes=_scenes())
    lane_change_plan = TrajectoryPipelinePlan(
        scenes=_scenes(lane_change=LaneChangeConfig(persist=1, negative_keep_every=1))
    )

    assert _run_pipeline(frame, plan=lane_change_plan, window_by="sequence") == _run_pipeline(
        frame, plan=standard_plan, window_by="sequence"
    )


def test_row_expanded_window_matches_legacy_list_window(scene_df_presets: DataFramePresets) -> None:
    frame = scene_df_presets["lane_change_sequences"]()

    actual = sliding_window(
        frame.lazy(), window_size=3, step_size=1, group_by="sequence", offset_sliding_col=True
    ).collect()
    expected = _legacy_sliding_window(
        frame.lazy(), window_size=3, step_size=1, group_by="sequence", offset_sliding_col=True
    ).collect()

    assert_frame_equal(actual, expected, check_row_order=True, check_column_order=True)


def test_batched_group_by_yield_matches_partition_by() -> None:
    frame = pl.DataFrame({
        "scene_id": [0, 0, 1, 1, 1, 2, 2],
        "frame": [0, 1, 0, 1, 2, 0, 1],
        "x": [0.0, 1.0, 10.0, 11.0, 12.0, 20.0, 21.0],
    })

    actual = [
        scene.collect().to_dict(as_series=False)
        for scene in tr.group_by_yield("scene_id", drop_group_cols=False, batch_size=2)(
            frame.lazy()
        )
    ]
    expected = [
        scene.to_dict(as_series=False)
        for scene in frame.partition_by("scene_id", maintain_order=True, as_dict=False)
    ]

    assert actual == expected


def _legacy_sliding_window(
    data: pl.LazyFrame,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    group_by: str | list[str] | None = None,
    offset_sliding_col: bool = False,
) -> pl.LazyFrame:
    group_keys = list(normalize_group_by(group_by))
    data = data.sort([*group_keys, sliding_col] if group_keys else sliding_col)

    slide_actual = pl.col(sliding_col)
    if offset_sliding_col:
        slide_actual -= slide_actual.first()

    grouped = data.group_by_dynamic(
        sliding_col, every=f"{step_size}i", period=f"{window_size}i", group_by=group_keys or None
    ).agg(slide_actual.alias(f"{sliding_col}_actual"), cs.all().exclude(sliding_col))

    span = (
        pl.col(f"{sliding_col}_actual").list.last()
        - pl.col(f"{sliding_col}_actual").list.first()
        + 1
    )
    return (
        grouped
        .filter(span == window_size)
        .with_row_index("window_index")
        .explode(cs.all().exclude("window_index", sliding_col, *group_keys))
        .drop(sliding_col)
        .rename({f"{sliding_col}_actual": sliding_col})
    )
