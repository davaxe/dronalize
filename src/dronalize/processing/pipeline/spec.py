"""Declarative configuration objects for trajectory processing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from typing_extensions import Self

from dronalize.processing.pipeline.extensions import LaneChangeSamplingExtension

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.processing.models import PipelinePlan
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

    plan: PipelinePlan
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


def _base_spec(
    plan: PipelinePlan, *, columns: TrackColumns | None, window_by: str | Sequence[str] | None
) -> TrajectorySpec:
    """Build the shared, extension-free portion of a trajectory spec."""
    return TrajectorySpec(plan=plan, columns=columns or TrackColumns(), window_by=window_by)


def standard(
    plan: PipelinePlan,
    *,
    columns: TrackColumns | None = None,
    window_by: str | Sequence[str] | None = None,
) -> TrajectorySpec:
    """Build the default trajectory pipeline specification."""
    return _base_spec(plan, columns=columns, window_by=window_by)


def lane_change_sampling(
    plan: PipelinePlan,
    *,
    lane_id: str = "lane_id",
    columns: TrackColumns | None = None,
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

    return base_spec.with_columns(base_spec.columns.with_lane_id(lane_id)).with_extension(
        LaneChangeSamplingExtension(
            negative_keep_every=sampling_config.negative_keep_every,
            min_lane_change_events=sampling_config.required_lane_changes,
            persist=sampling_config.persist,
            margin_before=sampling_config.margin_before,
            margin_after=sampling_config.margin_after,
        )
    )
