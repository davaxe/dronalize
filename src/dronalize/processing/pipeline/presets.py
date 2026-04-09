"""Convenience constructors for common trajectory pipeline specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.processing.pipeline.extensions import LaneChangeSamplingExtension
from dronalize.processing.pipeline.spec import TrackColumns, TrajectorySpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.processing.loading.config import LoaderConfig
    from dronalize.processing.loading.splits import SplitConfig


def _base_spec(
    config: LoaderConfig,
    *,
    split_request: SplitConfig | None,
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
    split_request: SplitConfig | None = None,
    columns: TrackColumns | None = None,
    window_by: str | Sequence[str] | None = None,
) -> TrajectorySpec:
    """Build the default trajectory pipeline specification."""
    return _base_spec(config, split_request=split_request, columns=columns, window_by=window_by)


def lane_change_sampling_spec(
    config: LoaderConfig,
    *,
    split_request: SplitConfig | None = None,
    lane_id: str = "lane_id",
    columns: TrackColumns | None = None,
    window_by: str | Sequence[str] | None = None,
) -> TrajectorySpec:
    """Build a spec for lane-change-aware window sampling."""
    sampling_config = config.lane_change_sampling
    base_spec = _base_spec(
        config, split_request=split_request, columns=columns, window_by=window_by
    )
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
