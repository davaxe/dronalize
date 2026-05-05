"""Small helpers for explicit dataset spec definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.config.models import (
    LaneChangeConfig,
    MinSamplesSpec,
    PruneByRuleSpec,
    ResampleConfig,
    ScenesConfig,
    ScreeningConfig,
    WindowConfig,
)
from dronalize.config.models.screening import ExcludeCategoriesSpec

if TYPE_CHECKING:
    from dronalize.config.base import ResampleMethod
    from dronalize.config.models.screening import AgentCheckSpec, CleanupSpec, SceneCheckSpec
    from dronalize.core.categories import AgentCategory


def combine_screenings(*screenings: ScreeningConfig) -> ScreeningConfig:
    """Combine multiple screenings into one by merging their cleanup rules."""
    combined_cleanup: dict[str, CleanupSpec] = {}
    combined_agent: dict[str, AgentCheckSpec] = {}
    combined_scene: dict[str, SceneCheckSpec] = {}
    for screening in screenings:
        if screening.cleanup:
            combined_cleanup.update(screening.cleanup)
        if screening.agent:
            combined_agent.update(screening.agent)
        if screening.scene:
            combined_scene.update(screening.scene)
    return ScreeningConfig(cleanup=combined_cleanup, agent=combined_agent, scene=combined_scene)


def exclude_category_screening(*category: AgentCategory) -> ScreeningConfig:
    """Return a cleanup-only screening that excludes the given agent categories."""
    return ScreeningConfig(cleanup={"category": ExcludeCategoriesSpec(categories=category)})


def minimum_samples_screening(minimum: int) -> ScreeningConfig:
    """Return the standard cleanup-only screening used by built-in dataset specs."""
    return ScreeningConfig(
        cleanup={"min_samples": PruneByRuleSpec(agent_rule=MinSamplesSpec(minimum=minimum))}
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


def spline_resample(
    up: int, down: int = 1, *, emit_velocity: bool = True, emit_acceleration: bool = True
) -> ResampleConfig:
    """Return the default cubic resampling config used by most trajectory datasets."""
    return resample_config(
        method="cubic",
        up=up,
        down=down,
        emit_velocity=emit_velocity,
        emit_acceleration=emit_acceleration,
    )


def scenes_config(
    *,
    history_frames: int,
    future_frames: int,
    sample_time: float,
    window_step: int | None = None,
    resample: ResampleConfig | None = None,
    lane_change: LaneChangeConfig | None = None,
) -> ScenesConfig:
    """Build an explicit `ScenesConfig` while keeping specs concise."""
    return ScenesConfig(
        history_frames=history_frames,
        future_frames=future_frames,
        sample_time=sample_time,
        window=WindowConfig(step=window_step) if window_step is not None else None,
        resample=resample,
        lane_change=lane_change,
    )
