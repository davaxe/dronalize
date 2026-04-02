"""Configuration models for loader-side preprocessing and ingestion."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self, TypedDict, Unpack

from dronalize.core.errors import LoaderConfigError
from dronalize.processing.filters.filter import Filter  # noqa: TC001
from dronalize.processing.pipeline.functional.resample import ResampleSpec  # noqa: TC001


class LoaderConfigUpdate(TypedDict, total=False):
    """Typed update payload for frozen `LoaderConfig` builder helpers."""

    input_len: int
    output_len: int
    sample_time: float
    resampling: ResampleSpec | None
    window: WindowConfig | None
    filter: Filter | None
    highway: HighwayParams | None


class WindowConfig(BaseModel):
    """Configuration for sliding window sampling of scenes."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    size: int = Field(gt=0)
    step: int = Field(gt=0)

    def __bool__(self) -> bool:
        """Return True if windowing is enabled."""
        return self.step > 0 and self.size > 0


class HighwayParams(BaseModel):
    """Configuration for highway-specific processing."""

    persist: int = Field(gt=0)
    margin_before: int = Field(default=0, ge=0)
    margin_after: int = Field(default=0, ge=0)
    required_lane_changes: int = Field(default=1, ge=0)
    negative_keep_every: int = Field(default=3, ge=1)


class LoaderConfig(BaseModel):
    """Base configuration class for trajectory data processing."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    input_len: int = Field(gt=0)
    output_len: int = Field(gt=0)
    sample_time: float = Field(gt=0)
    resampling: ResampleSpec | None = Field(default=None)
    window: WindowConfig | None = Field(default=None)
    filter: Filter | None = Field(default=None)
    highway: HighwayParams | None = Field(default=None)

    @model_validator(mode="after")
    def _validate(self) -> LoaderConfig:
        self._validate_window()
        return self

    def _validate_window(self) -> None:
        if self.window is None:
            return
        sequence_length = self._sequence_length()
        if self.window.size != sequence_length:
            msg = (
                f"Window size ({self.window.size}) must equal input_len + output_len "
                f"({sequence_length}) for consistent windowing."
            )
            raise LoaderConfigError(msg)

    def _sequence_length(self) -> int:
        """Return the total number of frames in one input/output sequence."""
        return self.input_len + self.output_len

    # -- builder helpers (return new frozen instances) -----------------------

    def with_window(self, step: int, size: int | None = None) -> Self:
        """Return a copy with the given window parameters.

        Parameters
        ----------
        step : int
            Number of frames to advance between consecutive windows.
        size : int, optional
            Total number of frames in each window. If None, defaults to
            `input_len + output_len`.

        Returns
        -------
        Self
            A **new** config instance with window parameters set.
        """
        new_window_params = WindowConfig(
            size=size if size is not None else self._sequence_length(), step=step
        )
        return self._updated(window=new_window_params)

    def with_filter(self, scene_filter: Filter | None) -> Self:
        """Return a copy with the given scene filter."""
        return self._updated(filter=scene_filter)

    def with_resampling(self, spec: ResampleSpec) -> Self:
        """Return a copy with the given resampling parameters.

        !!! note
            The sampling time in the provided spec will be overridden to match
            the config's `sample_time`.

        Parameters
        ----------
        spec : ResampleSpec
            Resampling specification.

        Returns
        -------
        Self
            A **new** config instance with resampling parameters set.
        """
        if spec.sample_time != self.sample_time:
            spec = spec.with_sample_time(self.sample_time)
        return self._updated(resampling=spec)

    def with_highway(
        self,
        persist: int = 1,
        margin_before: int = 0,
        margin_after: int = 0,
        required_lane_changes: int = 1,
        negative_keep_every: int = 3,
    ) -> Self:
        """Return a copy with the given highway parameters."""
        return self._updated(
            highway=HighwayParams(
                persist=persist,
                margin_before=margin_before,
                margin_after=margin_after,
                required_lane_changes=required_lane_changes,
                negative_keep_every=negative_keep_every,
            )
        )

    def _updated(self, **updates: Unpack[LoaderConfigUpdate]) -> Self:
        current = {name: getattr(self, name) for name in type(self).model_fields}
        return type(self).model_validate({**current, **updates})

    @property
    def resampled_input_len(self) -> int:
        """Observation length in frames after resampling."""
        if self.resampling is None:
            return self.input_len
        ratio = self.resampling.up / self.resampling.down
        return int((self.input_len - 1) * ratio + 1)

    @property
    def resampled_output_len(self) -> int:
        """Prediction length in frames after resampling."""
        if self.resampling is None:
            return self.output_len
        ratio = self.resampling.up / self.resampling.down
        total_len = int((self._sequence_length() - 1) * ratio + 1)
        return total_len - self.resampled_input_len

    @property
    def post_sample_time(self) -> float:
        """Time interval between frames after resampling."""
        if self.resampling is None:
            return self.sample_time
        ratio = self.resampling.up / self.resampling.down
        return self.sample_time / ratio
