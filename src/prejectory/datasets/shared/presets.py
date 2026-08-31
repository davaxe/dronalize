"""Small helpers for explicit dataset descriptor definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from prejectory.config.models import (
    LaneChangeConfig,
    PredictionTaskConfig,
    ResampleConfig,
    ScenesConfig,
    ScreeningConfig,
    WindowConfig,
)
from prejectory.datasets.registry import (
    DatasetTemporalSupport,
    DatasetWindowingSupport,
    FrameBounds,
)
from prejectory.processing.screening.agent import (
    AgentCheckRule,
    AgentRequireFrames,
    MinObservations,
)
from prejectory.processing.screening.base import PassingRequirement
from prejectory.processing.screening.cleanup import CleanupRule, ExcludeCategories, PruneByRule

if TYPE_CHECKING:
    from prejectory.config.base import ResampleMethod
    from prejectory.core.categories import AgentCategory
    from prejectory.datasets.registry import FrameBoundConfidence, SourceTemporalUnit
    from prejectory.processing.screening.scene import SceneCheckRule


def combine_screenings(*screenings: ScreeningConfig) -> ScreeningConfig:
    """Combine multiple screening configs by merging named rule maps."""
    combined_cleanup: dict[str, CleanupRule] = {}
    combined_agent: dict[str, AgentCheckRule] = {}
    combined_scene: dict[str, SceneCheckRule] = {}
    for screening in screenings:
        if screening.cleanup:
            combined_cleanup.update(screening.cleanup)
        if screening.agents:
            combined_agent.update(screening.agents)
        if screening.scenes:
            combined_scene.update(screening.scenes)
    return ScreeningConfig(cleanup=combined_cleanup, agents=combined_agent, scenes=combined_scene)


def exclude_category_screening(*category: AgentCategory) -> ScreeningConfig:
    """Return a cleanup-only screening that excludes the given agent categories."""
    return ScreeningConfig(cleanup={"category": ExcludeCategories.define(category)})


def minimum_observations_screening(
    minimum: int,
    *,
    required_frame: int | None = None,
) -> ScreeningConfig:
    """Return the standard screening used by built-in dataset descriptors.

    Parameters
    ----------
    minimum : int
        Minimum number of observations required for each agent to be retained in the
        scene. This will cleanup all agents that do not meet this requirement.
    required_frame : int | None, optional
        If not None, also require that each retained agent has a valid observation at
        the given frame index (relative to the start of the scene) in order for
        the scene to be retained. This rule requires only that at least one
        agent meets the requirement.

    Returns
    -------
    ScreeningConfig
        Screening config that applies the screening.

    """
    screening = ScreeningConfig(
        cleanup={"min_observations": PruneByRule(agent_rule=MinObservations(minimum=minimum))},
    )
    if required_frame is None:
        return screening
    if required_frame < 0:
        msg = "required_frame must be non-negative."
        raise ValueError(msg)
    return combine_screenings(
        screening,
        require_frames_screening(required_frame, require_absolute=1),
    )


def require_frames_screening(
    *frames: int,
    require_absolute: int | None = None,
    require_relative: float | None = None,
) -> ScreeningConfig:
    """Return an agent-screening config that requires specific frames."""
    require = (
        PassingRequirement(absolute=require_absolute, relative=require_relative)
        if require_absolute is not None or require_relative is not None
        else None
    )
    return ScreeningConfig(
        agents={"require_frames": AgentRequireFrames.define(frames, require=require)},
    )


def lane_change_sampling(
    *,
    persist: int = 2,
    margin_before: int = 0,
    margin_after: int = 0,
    required_lane_changes: int = 1,
    negative_keep_every: int = 3,
) -> LaneChangeConfig:
    """Return a lane-change sampling config with explicit defaults."""
    return LaneChangeConfig(
        persist=persist,
        margin_before=margin_before,
        margin_after=margin_after,
        required_lane_changes=required_lane_changes,
        negative_keep_every=negative_keep_every,
    )


def resample_config(
    *,
    method: ResampleMethod,
    up: int,
    down: int = 1,
    emit_velocity: bool = False,
    emit_acceleration: bool = False,
) -> ResampleConfig:
    """Return a temporal resampling config."""
    return ResampleConfig(
        method=method,
        up=up,
        down=down,
        emit_velocity=emit_velocity,
        emit_acceleration=emit_acceleration,
    )


def linear_resample(up: int, down: int = 1) -> ResampleConfig:
    """Return a linear resampling config."""
    return resample_config(method="linear", up=up, down=down)


def scenes_config(
    *,
    horizon_frames: int,
    sample_time: float,
    window_step: int | None = None,
    window_policy: Literal["strict", "anchored", "partial"] = "strict",
    resample: ResampleConfig | None = None,
    lane_change: LaneChangeConfig | None = None,
) -> ScenesConfig:
    """Build an explicit `ScenesConfig` while keeping descriptor defaults concise."""
    return ScenesConfig(
        horizon_frames=horizon_frames,
        sample_time=sample_time,
        window=WindowConfig(step=window_step, policy=window_policy)
        if window_step is not None
        else None,
        resample=resample,
        lane_change=lane_change,
    )


def benchmark_task(
    scenes: ScenesConfig,
    *,
    prediction_origin: int,
    prediction_end: int | None = None,
) -> PredictionTaskConfig:
    """Build a benchmark prediction task using a descriptor's scene horizon."""
    return PredictionTaskConfig(
        prediction_origin=prediction_origin,
        prediction_end=scenes.horizon_frames if prediction_end is None else prediction_end,
    )


def temporal_support(
    *,
    source_unit: SourceTemporalUnit,
    min_frames: int,
    max_frames: int,
    enabled_by_default: bool,
    confidence: FrameBoundConfidence = "observed",
) -> DatasetTemporalSupport:
    """Build standard temporal-support metadata from known source frame bounds."""
    return DatasetTemporalSupport(
        source_unit=source_unit,
        source_frame_bounds=FrameBounds(
            min_frames=min_frames,
            max_frames=max_frames,
            varying=min_frames != max_frames,
            confidence=confidence,
        ),
        windowing=DatasetWindowingSupport(
            enabled_by_default=enabled_by_default,
            default_policy="strict",
            supported_policies=("strict", "anchored", "partial"),
            max_window_frames=max_frames,
            validation="error",
        ),
    )
