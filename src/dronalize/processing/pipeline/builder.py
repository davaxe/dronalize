"""Structured builder for trajectory pipeline assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dronalize.core.polars_ops import normalize_group_by
from dronalize.processing.pipeline._internal import SCENE_ID_COLUMN, SPLIT_PARTITION_COLUMN
from dronalize.processing.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from dronalize.config.models import ScenesConfig
    from dronalize.processing.models import PipelinePlan, SplitRequest
    from dronalize.processing.pipeline.spec import TrackColumns, TrajectorySpec


@dataclass(slots=True)
class TrajectoryPipelineBuilder:
    """Mutable assembly state used by pipeline extensions."""

    spec: TrajectorySpec
    pre_window: Pipeline = field(default_factory=Pipeline)
    post_window: Pipeline = field(default_factory=Pipeline)
    post_screening: Pipeline = field(default_factory=Pipeline)

    @property
    def plan(self) -> PipelinePlan:
        """Return the pipeline plan for this pipeline."""
        return self.spec.plan

    @property
    def scenes(self) -> ScenesConfig:
        """Return the active scene-processing configuration."""
        return self.spec.plan.scenes

    @property
    def columns(self) -> TrackColumns:
        """Return configured trajectory column names."""
        return self.spec.columns

    @property
    def split_request(self) -> SplitRequest | None:
        """Return the active split request, if any."""
        return self.spec.plan.split

    @property
    def scene_id_column(self) -> str:
        """Return the internal scene-identity column name."""
        return SCENE_ID_COLUMN

    @property
    def has_window(self) -> bool:
        """Return whether the base pipeline will apply window extraction."""
        return self.scenes.window is not None

    def split_columns(self) -> list[str]:
        """Return grouping columns introduced by block-based splitting."""
        return (
            [SPLIT_PARTITION_COLUMN]
            if self.split_request is not None
            and self.split_request.strategy in {"time", "shuffled-time"}
            else []
        )

    def window_group_columns(self) -> list[str]:
        """Return grouping columns used while generating windows."""
        return [*normalize_group_by(self.spec.window_by), *self.split_columns()]

    def scene_key_columns(self) -> list[str]:
        """Return columns that uniquely identify one extracted scene."""
        group_columns = [*normalize_group_by(self.spec.window_by), *self.split_columns()]
        if self.has_window:
            group_columns.append("window_index")
        return group_columns

    def uses_scene_id(self) -> bool:
        """Return whether assembly needs an explicit scene identifier."""
        return len(self.scene_key_columns()) > 0

    def add_pre_window(self, pipeline: Pipeline) -> None:
        """Compose transforms before window extraction."""
        self.pre_window = self.pre_window.compose(pipeline)

    def add_post_window(self, pipeline: Pipeline) -> None:
        """Compose transforms immediately after window extraction."""
        self.post_window = self.post_window.compose(pipeline)

    def add_post_screening(self, pipeline: Pipeline) -> None:
        """Compose transforms after scene screening."""
        self.post_screening = self.post_screening.compose(pipeline)
