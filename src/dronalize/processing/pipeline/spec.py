"""Declarative configuration objects for trajectory processing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.processing.ingest.config import LoaderConfig
    from dronalize.processing.ingest.splits import SplitConfig
    from dronalize.processing.pipeline.extensions.base import TrajectoryPipelineExtension


@dataclass(frozen=True, slots=True)
class TrackColumns:
    """Column names used by the trajectory-processing pipeline."""

    frame: str = "frame"
    agent_id: str = "id"
    category: str = "agent_category"
    lane_id: str | None = None

    def with_lane_id(self, lane_id: str) -> Self:
        """Return a copy with the given lane-id column configured."""
        return replace(self, lane_id=lane_id)


@dataclass(frozen=True, slots=True)
class TrajectorySpec:
    """Declarative specification for building a trajectory-processing pipeline."""

    config: LoaderConfig
    split_request: SplitConfig | None = None
    columns: TrackColumns = field(default_factory=TrackColumns)
    window_by: str | Sequence[str] | None = None
    extension: TrajectoryPipelineExtension | None = None

    def with_columns(self, columns: TrackColumns) -> Self:
        """Return a copy with the given trajectory column mapping configured."""
        return replace(self, columns=columns)

    def with_window_by(self, window_by: str | Sequence[str] | None) -> Self:
        """Return a copy with the given pre-window grouping configured."""
        return replace(self, window_by=window_by)

    def with_extension(self, extension: TrajectoryPipelineExtension | None) -> Self:
        """Return a copy with the given pipeline extension configured."""
        return replace(self, extension=extension)
