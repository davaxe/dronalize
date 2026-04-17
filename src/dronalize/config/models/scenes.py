"""Configuration models for scene construction and processing."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from dronalize.config.base import FullConfig, PartialConfig, ResampleMethod
from dronalize.core.errors import ConfigurationError

Derivatives = Literal["velocity", "acceleration"]
"""Named derivative families that scene processing can materialize.

These labels are used by configuration helpers and schema-planning code to
describe which derived motion quantities should be emitted on top of sampled
positions.
"""


class ResampleConfig(FullConfig):
    """Validated specification for temporal resampling."""

    up: int = Field(default=1, gt=0)
    """Upsampling factor applied before downsampling."""
    down: int = Field(default=1, gt=0)
    """Downsampling factor applied after upsampling."""
    method: ResampleMethod = "linear"
    """Interpolation method used during resampling."""
    coordinates: tuple[str, ...] = Field(default=("x", "y"))
    """Coordinate fields resampled by the interpolation step."""
    emit_velocity: bool = Field(default=False)
    """Whether velocity derivatives should be emitted during resampling."""
    emit_acceleration: bool = Field(default=False)
    """Whether acceleration derivatives should be emitted during resampling."""
    max_gap: int = Field(default=1, gt=0)
    """Maximum consecutive missing frames allowed when interpolating samples."""

    @model_validator(mode="after")
    def _validate(self) -> ResampleConfig:
        if self.method == "linear" and (self.emit_velocity or self.emit_acceleration):
            msg = "Linear resampling does not support emitting derivatives."
            raise ConfigurationError(msg)
        return self


class PartialResampleConfig(PartialConfig[ResampleConfig]):
    """Patch model for partially overriding temporal resampling settings."""

    up: int | None = None
    """Replacement upsampling factor."""
    down: int | None = None
    """Replacement downsampling factor."""
    method: ResampleMethod | None = None
    """Replacement interpolation method."""
    coordinates: tuple[str, ...] | None = None
    """Replacement coordinate fields to resample."""
    emit_velocity: bool | None = None
    """Replacement policy for emitting velocity derivatives."""
    emit_acceleration: bool | None = None
    """Replacement policy for emitting acceleration derivatives."""
    max_gap: int | None = None
    """Replacement maximum consecutive gap allowed during interpolation."""
    full_config_type: type[ResampleConfig] = Field(default=ResampleConfig, init=False, repr=False)


class WindowConfig(FullConfig):
    """Configuration for sliding window sampling of scenes."""

    step: int = Field(gt=0)
    """Stride between consecutive sampled windows in frames."""


class PartialWindowConfig(PartialConfig[WindowConfig]):
    """Patch model for partially overriding sliding-window sampling settings."""

    step: int | None = None
    """Replacement stride between consecutive sampled windows in frames."""
    full_config_type: type[WindowConfig] = Field(default=WindowConfig, init=False, repr=False)


class LaneChangeConfig(FullConfig):
    """Configuration for lane-change-aware sampling."""

    persist: int = Field(gt=0)
    """Minimum number of frames a lane-change state must persist to count."""
    margin_before: int = Field(default=0, ge=0)
    """Extra context frames to keep before the detected lane change."""
    margin_after: int = Field(default=0, ge=0)
    """Extra context frames to keep after the detected lane change."""
    required_lane_changes: int = Field(default=1, ge=0)
    """Minimum number of lane changes required for a positive sample."""
    negative_keep_every: int = Field(default=3, ge=1)
    """Keep one negative sample out of every N candidates."""


class PartialLaneChangeConfig(PartialConfig[LaneChangeConfig]):
    """Patch model for partially overriding lane-change-aware sampling settings."""

    persist: int | None = None
    """Replacement persistence threshold for lane-change detection."""
    margin_before: int | None = None
    """Replacement context margin before a detected lane change."""
    margin_after: int | None = None
    """Replacement context margin after a detected lane change."""
    required_lane_changes: int | None = None
    """Replacement minimum lane-change count for positive samples."""
    negative_keep_every: int | None = None
    """Replacement negative-sample retention interval."""
    full_config_type: type[LaneChangeConfig] = Field(
        default=LaneChangeConfig, init=False, repr=False
    )


class ScenesConfig(FullConfig):
    """Base configuration class for scene construction and temporal transforms."""

    history_frames: int = Field(gt=0)
    """Number of history frames included in each scene."""
    future_frames: int = Field(gt=0)
    """Number of prediction frames included in each scene."""
    sample_time: float = Field(gt=0)
    """Time interval between consecutive frames in seconds."""
    window: WindowConfig | None = Field(default=None)
    """Optional sliding-window sampling configuration."""
    resample: ResampleConfig | None = Field(default=None)
    """Optional temporal resampling configuration applied before scene emission."""
    lane_change: LaneChangeConfig | None = Field(default=None)
    """Optional lane-change-aware sampling configuration."""


class PartialScenesConfig(PartialConfig[ScenesConfig]):
    """Patch model for partially overriding scene construction settings."""

    history_frames: int | None = None
    """Replacement number of history frames per scene."""
    future_frames: int | None = None
    """Replacement number of prediction frames per scene."""
    sample_time: float | None = None
    """Replacement frame interval in seconds."""
    window: PartialWindowConfig | None = None
    """Partial override for sliding-window sampling settings."""
    resample: PartialResampleConfig | None = None
    """Partial override for temporal resampling settings."""
    lane_change: PartialLaneChangeConfig | None = None
    """Partial override for lane-change-aware sampling settings."""
    full_config_type: type[ScenesConfig] = Field(default=ScenesConfig, init=False, repr=False)


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
