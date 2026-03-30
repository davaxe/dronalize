"""Convenience constructors for common trajectory pipeline specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.processing.pipeline.extensions import LaneChangeSamplingExtension
from dronalize.processing.pipeline.spec import LaneChangeDetection, TrackColumns, TrajectorySpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.processing.ingest.config import LoaderConfig
    from dronalize.processing.ingest.splits import SplitRequest


def _base_spec(
    config: LoaderConfig,
    *,
    split_request: SplitRequest | None,
    columns: TrackColumns | None,
    window_by: str | Sequence[str] | None,
) -> TrajectorySpec:
    """Build the shared, extension-free portion of a trajectory spec."""
    return TrajectorySpec(
        config=config,
        split_request=split_request,
        columns=columns or TrackColumns(),
        window_by=window_by,
    )


def standard_trajectory_spec(
    config: LoaderConfig,
    *,
    split_request: SplitRequest | None = None,
    columns: TrackColumns | None = None,
    window_by: str | Sequence[str] | None = None,
) -> TrajectorySpec:
    """Build the default trajectory pipeline specification."""
    return _base_spec(config, split_request=split_request, columns=columns, window_by=window_by)


def highway_trajectory_spec(
    config: LoaderConfig,
    *,
    split_request: SplitRequest | None = None,
    lane_id: str = "lane_id",
    negative_keep_every: int = 3,
    min_lane_change_events: int = 1,
    lane_change: LaneChangeDetection | None = None,
    columns: TrackColumns | None = None,
    window_by: str | Sequence[str] | None = None,
) -> TrajectorySpec:
    """Build a spec for lane-change-aware highway window sampling."""
    lane_change = lane_change or LaneChangeDetection()
    base_spec = _base_spec(
        config, split_request=split_request, columns=columns, window_by=window_by
    )
    return base_spec.with_columns(base_spec.columns.with_lane_id(lane_id)).with_extension(
        LaneChangeSamplingExtension(
            negative_keep_every=negative_keep_every,
            min_lane_change_events=min_lane_change_events,
            persist=lane_change.persist,
            margin_before=lane_change.margin_before,
            margin_after=lane_change.margin_after,
        )
    )
