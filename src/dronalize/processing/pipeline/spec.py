"""Declarative configuration objects for trajectory processing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

from typing_extensions import Self

from dronalize.core.functional.basic import normalize_group_by
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.pipeline import stages
from dronalize.processing.pipeline.context import BuildContext
from dronalize.processing.pipeline.contributions import (
    PipelineStage,
    StageContributions,
    get_stage_pipelines,
)
from dronalize.processing.pipeline.extensions.lane_change import LaneChangeSamplingExtension
from dronalize.processing.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.processing.models import PipelinePlan


SPLIT_PARTITION_COLUMN = "_split_partition"
SCENE_ID_COLUMN = "_scene_id"
WINDOW_INDEX_COLUMN = "window_index"


class TrajectoryPipelineExtension(Protocol):
    """Protocol for extensions that contribute stage-local pipeline transforms."""

    def compile(self, ctx: BuildContext) -> StageContributions:
        """Compile this extension against an immutable build context."""
        ...


@dataclass(frozen=True, slots=True)
class TrajectorySpec:
    """Declarative specification for building a trajectory-processing pipeline."""

    plan: PipelinePlan
    columns: TrajectoryColumns = field(default_factory=TrajectoryColumns)
    window_by: str | Sequence[str] | None = None
    extension: TrajectoryPipelineExtension | None = None

    def with_window_by(self, window_by: str | Sequence[str] | None) -> Self:
        """Return a copy with the given pre-window grouping configured."""
        return replace(self, window_by=window_by)

    def with_extension(self, extension: TrajectoryPipelineExtension | None) -> Self:
        """Return a copy with the given pipeline extension configured."""
        return replace(self, extension=extension)

    @property
    def frame_column(self) -> str:
        """Return the name of the frame column."""
        return self.columns.frame

    @property
    def agent_id_column(self) -> str:
        """Return the name of the agent ID column."""
        return self.columns.agent_id

    @property
    def category_column(self) -> str:
        """Return the name of the agent category column."""
        return self.columns.category


def compile_build_context(spec: TrajectorySpec, *, require_scene_id: bool = False) -> BuildContext:
    """Compile immutable derived pipeline state from a declarative spec."""
    plan = spec.plan
    scenes = plan.scenes
    assignment_request = plan.assignment
    has_window = scenes.window is not None
    split_columns = (
        (SPLIT_PARTITION_COLUMN,)
        if assignment_request is not None
        and assignment_request.strategy in {"time", "shuffled-time"}
        else ()
    )
    window_group_columns = (*normalize_group_by(spec.window_by), *split_columns)
    scene_key_columns = (
        (*window_group_columns, WINDOW_INDEX_COLUMN) if has_window else window_group_columns
    )
    scene_id_column = SCENE_ID_COLUMN if (scene_key_columns or require_scene_id) else None

    drop_columns = (
        *((WINDOW_INDEX_COLUMN,) if has_window else ()),
        *split_columns,
        *((scene_id_column,) if scene_id_column is not None else ()),
    )

    return BuildContext(
        plan=plan,
        columns=spec.columns,
        window_by=tuple(normalize_group_by(spec.window_by)),
        split_columns=split_columns,
        window_group_columns=window_group_columns,
        scene_key_columns=scene_key_columns,
        scene_id_column=scene_id_column,
        drop_columns=drop_columns,
    )


def trajectory_pipeline(spec: TrajectorySpec) -> Pipeline:
    """Build a trajectory-processing pipeline from a declarative spec."""
    initial_ctx = compile_build_context(spec, require_scene_id=False)
    contributions = (
        spec.extension.compile(initial_ctx) if spec.extension is not None else StageContributions()
    )
    ctx = compile_build_context(spec, require_scene_id=contributions.require_scene_id)
    return (
        Pipeline()
        >> stages.build_split_stage(ctx)
        >> get_stage_pipelines(contributions, PipelineStage.PRE_WINDOW)
        >> stages.build_window_stage(ctx)
        >> get_stage_pipelines(contributions, PipelineStage.POST_WINDOW)
        >> stages.build_scene_id_stage(ctx)
        >> stages.build_screening_stage(ctx)
        >> get_stage_pipelines(contributions, PipelineStage.POST_SCREENING)
        >> stages.build_output_stage(ctx)
    )


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
