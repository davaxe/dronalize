"""Pipeline assembly helpers for trajectory processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.pipeline.contributions import (
    PipelineStage,
    StageContributions,
    get_stage_pipelines,
)
from dronalize.processing.pipeline.extensions.lane_change import LaneChangeSamplingExtension
from dronalize.processing.pipeline.pipeline import Pipeline
from dronalize.processing.pipeline.spec import TrajectorySpec, compile_build_context
from dronalize.processing.pipeline.stages import (
    build_output_stage,
    build_scene_id_stage,
    build_screening_stage,
    build_split_stage,
    build_window_stage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.processing.models import PipelinePlan


def trajectory_pipeline(spec: TrajectorySpec) -> Pipeline:
    """Build a trajectory-processing pipeline from a declarative spec."""
    initial_ctx = compile_build_context(spec, require_scene_id=False)
    contributions = (
        spec.extension.compile(initial_ctx) if spec.extension is not None else StageContributions()
    )
    ctx = compile_build_context(spec, require_scene_id=contributions.require_scene_id)

    pipeline = Pipeline()
    pipeline = pipeline.compose(build_split_stage(ctx))

    for stage_pipeline in get_stage_pipelines(contributions, PipelineStage.PRE_WINDOW):
        pipeline = pipeline.compose(stage_pipeline)

    pipeline = pipeline.compose(build_window_stage(ctx))

    for stage_pipeline in get_stage_pipelines(contributions, PipelineStage.POST_WINDOW):
        pipeline = pipeline.compose(stage_pipeline)

    pipeline = pipeline.compose(build_scene_id_stage(ctx))
    pipeline = pipeline.compose(build_screening_stage(ctx))

    for stage_pipeline in get_stage_pipelines(contributions, PipelineStage.POST_SCREENING):
        pipeline = pipeline.compose(stage_pipeline)

    return pipeline.compose(build_output_stage(ctx))


def standard(
    plan: PipelinePlan,
    *,
    columns: TrajectoryColumns | None = None,
    window_by: str | Sequence[str] | None = None,
) -> TrajectorySpec:
    """Build the default trajectory pipeline specification."""
    return _base_spec(plan, columns=columns, window_by=window_by)


def lane_change_sampling(
    plan: PipelinePlan,
    *,
    columns: TrajectoryColumns | None = None,
    lane_id: str = "lane_id",
    window_by: str | Sequence[str] | None = None,
) -> TrajectorySpec:
    """Build a spec for lane-change-aware window sampling."""
    sampling_config = plan.scenes.lane_change
    base_spec = _base_spec(plan, columns=columns, window_by=window_by)
    if sampling_config is None or sampling_config.negative_keep_every == 1:
        # If every negative sample is kept, there is no difference in output
        # selection between lane-change-aware sampling and standard sampling, so
        # skip the extension (which is more efficient).
        return base_spec

    return base_spec.with_extension(
        LaneChangeSamplingExtension(
            lane_id_column=lane_id,
            negative_keep_every=sampling_config.negative_keep_every,
            min_lane_change_events=sampling_config.required_lane_changes,
            persist=sampling_config.persist,
            margin_before=sampling_config.margin_before,
            margin_after=sampling_config.margin_after,
        )
    )


def _base_spec(
    plan: PipelinePlan,
    *,
    columns: TrajectoryColumns | None = None,
    window_by: str | Sequence[str] | None,
) -> TrajectorySpec:
    """Build the shared, extension-free portion of a trajectory spec."""
    return TrajectorySpec(
        plan=plan, columns=TrajectoryColumns() if columns is None else columns, window_by=window_by
    )
