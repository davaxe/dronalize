from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from dronalize.config.base import FullConfig, PartialConfig, ResampleMethod
from dronalize.core.errors import ConfigurationError

Derivatives = Literal["velocity", "acceleration"]


class ResampleConfig(FullConfig):
    """Validated specification for temporal resampling."""

    up: int = Field(default=1, gt=0)
    down: int = Field(default=1, gt=0)
    method: ResampleMethod = "linear"
    coordinates: tuple[str, ...] = Field(default=("x", "y"))
    emit_velocity: bool = Field(default=False)
    emit_acceleration: bool = Field(default=False)
    max_gap: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def _validate(self) -> ResampleConfig:
        if self.method == "linear" and (self.emit_velocity or self.emit_acceleration):
            msg = "Linear resampling does not support emitting derivatives."
            raise ConfigurationError(msg)
        return self


class PartialResampleConfig(PartialConfig[ResampleConfig]):
    up: int | None = None
    down: int | None = None
    method: ResampleMethod | None = None
    coordinates: tuple[str, ...] | None = None
    emit_velocity: bool | None = None
    emit_acceleration: bool | None = None
    max_gap: int | None = None
    full_config_type: type[ResampleConfig] = ResampleConfig


class WindowConfig(FullConfig):
    """Configuration for sliding window sampling of scenes."""

    step: int = Field(gt=0)


class PartialWindowConfig(PartialConfig[WindowConfig]):
    step: int | None = None
    full_config_type: type[WindowConfig] = WindowConfig


class LaneChangeConfig(FullConfig):
    """Configuration for lane-change-aware sampling."""

    persist: int = Field(gt=0)
    margin_before: int = Field(default=0, ge=0)
    margin_after: int = Field(default=0, ge=0)
    required_lane_changes: int = Field(default=1, ge=0)
    negative_keep_every: int = Field(default=3, ge=1)


class PartialLaneChangeConfig(PartialConfig[LaneChangeConfig]):
    persist: int | None = None
    margin_before: int | None = None
    margin_after: int | None = None
    required_lane_changes: int | None = None
    negative_keep_every: int | None = None
    full_config_type: type[LaneChangeConfig] = LaneChangeConfig


class ScenesConfig(FullConfig):
    """Base configuration class for scene construction and temporal transforms."""

    history_frames: int = Field(gt=0)
    future_frames: int = Field(gt=0)
    sample_time: float = Field(gt=0)
    window: WindowConfig | None = Field(default=None)
    resample: ResampleConfig | None = Field(default=None)
    lane_change: LaneChangeConfig | None = Field(default=None)


class PartialScenesConfig(PartialConfig[ScenesConfig]):
    history_frames: int | None = None
    future_frames: int | None = None
    sample_time: float | None = None
    window: PartialWindowConfig | None = None
    resample: PartialResampleConfig | None = None
    lane_change: PartialLaneChangeConfig | None = None
    full_config_type: type[ScenesConfig] = ScenesConfig


def effective_scene_window(config: ScenesConfig) -> tuple[int, int, float]:
    """Return history frames, future frames, and sample time after resampling."""
    if config.resample is None:
        return config.history_frames, config.future_frames, config.sample_time

    up = config.resample.up
    down = config.resample.down
    ratio = up / down
    total_len = config.history_frames + config.future_frames
    total_resampled_len = int((total_len - 1) * ratio + 1)
    history_resampled = int((config.history_frames - 1) * ratio + 1)
    future_resampled = total_resampled_len - history_resampled
    return (history_resampled, future_resampled, config.sample_time * down / up)
