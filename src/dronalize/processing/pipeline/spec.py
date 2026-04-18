"""Declarative configuration objects for trajectory processing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

from typing_extensions import Self

from dronalize.core.polars_ops import normalize_group_by
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.pipeline._internal import (
    SCENE_ID_COLUMN,
    SPLIT_PARTITION_COLUMN,
    WINDOW_INDEX_COLUMN,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.config.models import ScenesConfig
    from dronalize.processing.models import PipelinePlan, SplitRequest
    from dronalize.processing.pipeline.contributions import StageContributions


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Execution-ready context with all derived pipeline state precompiled."""

    spec: TrajectorySpec
    split_columns: tuple[str, ...]
    window_group_columns: tuple[str, ...]
    scene_key_columns: tuple[str, ...]
    scene_id_column: str | None
    drop_columns: tuple[str, ...]

    @property
    def frame_column(self) -> str:
        """Return the frame column configured by the trajectory spec."""
        return self.spec.frame_column

    @property
    def agent_id_column(self) -> str:
        """Return the agent-ID column configured by the trajectory spec."""
        return self.spec.agent_id_column

    @property
    def category_column(self) -> str:
        """Return the category column configured by the trajectory spec."""
        return self.spec.category_column

    @property
    def has_window(self) -> bool:
        """Return whether the pipeline will build sliding-window scenes."""
        return self.scenes.window is not None

    @property
    def split_request(self) -> SplitRequest | None:
        """Return the active split request, if the plan uses one."""
        return self.plan.split

    @property
    def scenes(self) -> ScenesConfig:
        """Return the scene-construction config from the pipeline plan."""
        return self.plan.scenes

    @property
    def plan(self) -> PipelinePlan:
        """Return the pipeline plan owned by this build context."""
        return self.spec.plan


def compile_build_context(spec: TrajectorySpec, *, require_scene_id: bool = False) -> BuildContext:
    """Compile immutable derived pipeline state from a declarative spec."""
    plan = spec.plan
    scenes = plan.scenes
    split_request = plan.split
    has_window = scenes.window is not None
    split_columns = (
        (SPLIT_PARTITION_COLUMN,)
        if split_request is not None and split_request.strategy in {"time", "shuffled-time"}
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
        spec=spec,
        split_columns=split_columns,
        window_group_columns=window_group_columns,
        scene_key_columns=scene_key_columns,
        scene_id_column=scene_id_column,
        drop_columns=drop_columns,
    )


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
