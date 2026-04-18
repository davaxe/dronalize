from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

import dronalize.processing.pipeline.transforms as pipeline_transforms
from dronalize.config.models import (
    LaneChangeConfig,
    ScenesConfig,
    SplitWeights,
    TimeSplitConfig,
    WindowConfig,
)
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.models import LoaderRequest, PipelinePlan, SplitRequest
from dronalize.processing.pipeline.contributions import PipelineStage, get_stage_pipelines
from dronalize.processing.pipeline.extensions.lane_change import LaneChangeSamplingExtension
from dronalize.processing.pipeline.factory import standard, trajectory_pipeline
from dronalize.processing.pipeline.spec import TrajectorySpec, compile_build_context
from dronalize.processing.pipeline.stages import build_output_stage

if TYPE_CHECKING:
    from collections.abc import Callable

from tests.support import DemoLoader

SCENE_ID_COLUMN = "_scene_id"
SPLIT_PARTITION_COLUMN = "_split_partition"
WINDOW_INDEX_COLUMN = "window_index"


def _scenes(
    *,
    history_frames: int = 2,
    future_frames: int = 1,
    window_step: int | None = None,
    lane_change: LaneChangeConfig | None = None,
) -> ScenesConfig:
    return ScenesConfig(
        history_frames=history_frames,
        future_frames=future_frames,
        sample_time=1.0,
        window=WindowConfig(step=window_step) if window_step is not None else None,
        lane_change=lane_change,
    )


def _plan(*, scenes: ScenesConfig | None = None, split: SplitRequest | None = None) -> PipelinePlan:
    return PipelinePlan(scenes=scenes or _scenes(), split=split)


def _trajectory_frame(
    *,
    frame_column: str = "frame",
    agent_id_column: str = "id",
    category_column: str = "agent_category",
    frames: range | list[int] | None = None,
) -> pl.DataFrame:
    frames = frames if frames is not None else range(3)
    frame_values = list(frames)
    return pl.DataFrame({
        frame_column: frame_values,
        agent_id_column: [1] * len(frame_values),
        category_column: [1] * len(frame_values),
        "x": [float(value) for value in frame_values],
        "y": [0.0] * len(frame_values),
    })


def test_standard_pipeline_uses_configured_columns_and_skips_scene_id() -> None:
    spec = standard(
        _plan(), columns=TrajectoryColumns(frame="f", agent_id="agent", category="category")
    )

    ctx = compile_build_context(spec)
    result = list(
        trajectory_pipeline(spec).execute(
            _trajectory_frame(
                frame_column="f", agent_id_column="agent", category_column="category"
            ),
            collect=True,
        )
    )

    assert len(result) == 1
    assert ctx.scene_id_column is None
    assert ctx.drop_columns == ()
    assert result is not None
    assert result[0].columns == ["f", "agent", "category", "x", "y"]
    assert SCENE_ID_COLUMN not in result[0].columns


def test_base_scene_loader_constructs_explicit_columns_from_native_schema() -> None:
    loader = DemoLoader(data_root=".", request=LoaderRequest(scenes=_scenes(window_step=1)))

    assert TrajectoryColumns.from_schema(loader.native_trajectory_schema()) == TrajectoryColumns(
        frame="frame",
        agent_id="id",
        category="agent_category",
        x="x",
        y="y",
        vx="vx",
        vy="vy",
        ax="ax",
        ay="ay",
        yaw="yaw",
    )


def test_windowed_pipeline_uses_window_index_for_scene_identity_and_drops_internal_columns() -> (
    None
):
    spec = standard(_plan(scenes=_scenes(window_step=1)))

    ctx = compile_build_context(spec)
    outputs = list(
        trajectory_pipeline(spec).execute(_trajectory_frame(frames=range(4)), collect=True)
    )

    assert ctx.scene_key_columns == (WINDOW_INDEX_COLUMN,)
    assert ctx.scene_id_column == SCENE_ID_COLUMN
    assert len(outputs) == 2
    assert all(WINDOW_INDEX_COLUMN not in output.columns for output in outputs)
    assert all(SCENE_ID_COLUMN not in output.columns for output in outputs)
    assert [output["frame"].to_list() for output in outputs] == [[0, 1, 2], [0, 1, 2]]


def test_time_split_pipeline_uses_split_partition_and_drops_generated_columns() -> None:
    split_request = SplitRequest(config=TimeSplitConfig(ratio=SplitWeights(train=0.5, val=0.5)))
    spec = standard(_plan(split=split_request))

    ctx = compile_build_context(spec)
    outputs = list(
        trajectory_pipeline(spec).execute(_trajectory_frame(frames=range(8)), collect=True)
    )

    assert ctx.window_group_columns == (SPLIT_PARTITION_COLUMN,)
    assert ctx.scene_key_columns == (SPLIT_PARTITION_COLUMN,)
    assert ctx.drop_columns == (SPLIT_PARTITION_COLUMN, SCENE_ID_COLUMN)
    assert outputs
    assert all("split" in output.columns for output in outputs)
    assert all(SPLIT_PARTITION_COLUMN not in output.columns for output in outputs)
    assert all(SCENE_ID_COLUMN not in output.columns for output in outputs)


def test_lane_change_extension_requests_scene_id_and_creates_pre_window_column() -> None:
    extension = LaneChangeSamplingExtension(
        lane_id_column="lane_id", persist=1, negative_keep_every=2, min_lane_change_events=1
    )
    spec = TrajectorySpec(plan=_plan(scenes=_scenes(window_step=1)), extension=extension)

    contributions = extension.compile(compile_build_context(spec))
    pre_stage = get_stage_pipelines(contributions, PipelineStage.PRE_WINDOW)

    result = list(
        pre_stage[0].execute(
            _trajectory_frame(frames=range(4)).with_columns(pl.Series("lane_id", [1, 1, 2, 2])),
            collect=True,
        )
    )

    assert len(result) == 1
    assert contributions.require_scene_id is True
    assert len(pre_stage) == 1
    assert result is not None
    assert "valid_lane_change" in result[0].columns


def test_lane_change_extension_samples_scenes_and_removes_temporary_columns() -> None:
    extension = LaneChangeSamplingExtension(
        persist=1, negative_keep_every=2, min_lane_change_events=1
    )
    spec = TrajectorySpec(plan=_plan(scenes=_scenes(window_step=1)), extension=extension)

    contributions = extension.compile(compile_build_context(spec))
    post_stage = get_stage_pipelines(contributions, PipelineStage.POST_SCREENING)
    result = list(
        post_stage[0].execute(
            pl.DataFrame({
                "frame": [0, 1, 0, 1],
                "id": [1, 1, 1, 1],
                "agent_category": [1, 1, 1, 1],
                "x": [0.0, 1.0, 0.0, 1.0],
                "y": [0.0, 0.0, 0.0, 0.0],
                SCENE_ID_COLUMN: [0, 0, 1, 1],
                "valid_lane_change": [True, False, False, False],
            }),
            collect=True,
        )
    )

    assert len(result) == 1

    assert len(post_stage) == 1
    assert result is not None
    assert result[0][SCENE_ID_COLUMN].to_list() == [0, 0]
    assert "valid_lane_change" not in result[0].columns
    assert "_scene_lane_change_count" not in result[0].columns


@pytest.mark.parametrize(
    ("spec", "expected_group_by"),
    [
        (standard(_plan(scenes=_scenes(window_step=1))), [SCENE_ID_COLUMN, "id"]),
        (standard(_plan()), ["id"]),
    ],
)
def test_build_output_stage_resample_grouping_depends_on_scene_id(
    monkeypatch: pytest.MonkeyPatch, spec: TrajectorySpec, expected_group_by: list[str]
) -> None:
    recorded: dict[str, object] = {}

    def _fake_resample(
        spec: object | None = None,
        *,
        frame_column: str = "frame",
        group_by: str | list[str] | None = None,
    ) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
        recorded["spec"] = spec
        recorded["frame_column"] = frame_column
        recorded["group_by"] = group_by

        def _identity(df: pl.LazyFrame) -> pl.LazyFrame:
            return df

        return _identity

    monkeypatch.setattr(pipeline_transforms, "resample", _fake_resample)

    _ = build_output_stage(compile_build_context(spec))

    assert recorded["frame_column"] == spec.frame_column
    assert recorded["group_by"] == expected_group_by
