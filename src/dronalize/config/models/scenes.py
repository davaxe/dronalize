"""Configuration models for scene construction and processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import Field, model_validator
from typing_extensions import override

from dronalize.config.base import ConfigBase, ConfigPatch, ResampleMethod, ResolvedConfig
from dronalize.core.errors import ConfigurationError

if TYPE_CHECKING:
    from dronalize.core.typing import T


class ResampleConfig(ResolvedConfig):
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


class PartialResampleConfig(ConfigPatch[ResampleConfig]):
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


class WindowConfig(ResolvedConfig):
    """Configuration for sliding window sampling of scenes."""

    step: int = Field(gt=0)
    """Stride between consecutive sampled windows in frames."""


class PartialWindowConfig(ConfigPatch[WindowConfig]):
    """Patch model for partially overriding sliding-window sampling settings."""

    step: int | None = None
    """Replacement stride between consecutive sampled windows in frames."""
    full_config_type: type[WindowConfig] = Field(default=WindowConfig, init=False, repr=False)


class LaneChangeConfig(ResolvedConfig):
    """Configuration for lane-change-aware sampling."""

    persist: int = Field(gt=0)
    """Minimum number of frames a lane-change state must persist to count."""
    margin_before: int = Field(default=0, ge=0)
    """Extra context frames to keep before the detected lane change."""
    margin_after: int = Field(default=0, ge=0)
    """Extra context frames to keep after the detected lane change."""
    required_lane_changes: int = Field(default=1, gt=0)
    """Minimum number of lane changes required for a positive sample."""
    negative_keep_every: int = Field(default=3, ge=1)
    """Keep one negative sample out of every N candidates."""


class PartialLaneChangeConfig(ConfigPatch[LaneChangeConfig]):
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


class ScenesConfig(ResolvedConfig):
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


class PartialScenesConfig(ConfigPatch[ScenesConfig]):
    """Patch model for partially overriding scene construction settings."""

    history_frames: int | None = None
    """Replacement number of history frames per scene."""
    future_frames: int | None = None
    """Replacement number of prediction frames per scene."""
    sample_time: float | None = None
    """Replacement frame interval in seconds."""
    window: PartialWindowConfig | Literal[False] | None = None
    """Partial override for sliding-window sampling settings."""
    resample: PartialResampleConfig | Literal[False] | None = None
    """Partial override for temporal resampling settings."""
    lane_change: PartialLaneChangeConfig | Literal[False] | None = None
    """Partial override for lane-change-aware sampling settings."""
    full_config_type: type[ScenesConfig] = Field(default=ScenesConfig, init=False, repr=False)

    @override
    def merge_into(self, target: ScenesConfig | None, *, exclude_none: bool = True) -> ScenesConfig:
        """Apply this partial scenes config to an existing full scenes config."""
        return ScenesConfig(
            history_frames=_resolve_required(
                "history_frames",
                self.history_frames,
                target.history_frames if target is not None else None,
            ),
            future_frames=_resolve_required(
                "future_frames",
                self.future_frames,
                target.future_frames if target is not None else None,
            ),
            sample_time=_resolve_required(
                "sample_time", self.sample_time, target.sample_time if target is not None else None
            ),
            window=_apply_optional_block(
                self.window, target.window if target is not None else None
            ),
            resample=_apply_optional_block(
                self.resample, target.resample if target is not None else None
            ),
            lane_change=_apply_optional_block(
                self.lane_change, target.lane_change if target is not None else None
            ),
        )


def _resolve_required(name: str, value: T | None, fallback: T | None) -> T:
    result = value if value is not None else fallback
    if result is None:
        msg = f"Missing required field: {name}"
        raise ValueError(msg)
    return result


ConfigT = TypeVar("ConfigT", bound=ConfigBase)


def _apply_optional_block(
    patch: ConfigPatch[ConfigT] | Literal[False] | None, target: ConfigT | None
) -> ConfigT | None:
    """Apply a patch to an optional nested config block."""
    if patch is None:
        return target
    if patch is False:
        return None
    return patch.merge_into(target)


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
