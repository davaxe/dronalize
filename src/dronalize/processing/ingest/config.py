from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dronalize.processing.filters.filter import Filter, normalize_filter_frames
from dronalize.processing.pipeline.functional.resample import ResampleSpec  # noqa: TC001


class WindowParams(BaseModel):
    """Configuration for sliding window sampling of scenes."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    window_size: int = Field(gt=0, description="Number of frames in each window.")
    step_size: int = Field(gt=0, description="Number of frames to skip between windows.")

    def __bool__(self) -> bool:
        """Return True if windowing is enabled (i.e., step_size > 0)."""
        return self.step_size > 0 and self.window_size > 0


class LoaderConfig(BaseModel):
    """Base configuration class for trajectory data processing."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    input_len: int = Field(gt=0)
    output_len: int = Field(gt=0)
    sample_time: float = Field(gt=0)
    resampling: ResampleSpec | None = Field(default=None)
    window: WindowParams | None = Field(default=None)
    filters: Filter | None = Field(default=None)
    extra_kwargs: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> LoaderConfig:
        self._validate_window()
        return self._normalize_filters()

    def _validate_window(self) -> None:
        if self.window is None:
            return
        sequence_length = self._sequence_length()
        if self.window.window_size != sequence_length:
            msg = (
                f"Window size ({self.window.window_size}) must equal input_len + output_len "
                f"({sequence_length}) for consistent windowing."
            )
            raise ValueError(msg)

    def _normalize_filters(self) -> LoaderConfig:
        if self.filters is None:
            return self

        normalized = normalize_filter_frames(
            self.filters,
            sequence_length=self._sequence_length(),
        )
        if normalized == self.filters:
            return self
        return self.model_copy(update={"filters": normalized})

    def _sequence_length(self) -> int:
        """Return the total number of frames in one input/output sequence."""
        return self.input_len + self.output_len

    # -- builder helpers (return new frozen instances) -----------------------

    def with_window(self, step_size: int, window_size: int | None = None) -> Self:
        """Return a copy with the given window parameters.

        Parameters
        ----------
        step_size : int
            Number of frames to advance between consecutive windows.
        window_size : int, optional
            Total number of frames in each window. If None, defaults to
            `input_len + output_len`.

        Returns
        -------
        Self
            A **new** config instance with window parameters set.
        """
        new_window_params = WindowParams(
            window_size=window_size if window_size is not None else self._sequence_length(),
            step_size=step_size,
        )
        return self._updated(window=new_window_params)

    def with_filters(self, filters: Filter | None) -> Self:
        """Return a copy with the given scene filters."""
        return self._updated(filters=filters)

    def with_resampling(
        self,
        spec: ResampleSpec,
    ) -> Self:
        """Return a copy with the given resampling parameters.

        Parameters
        ----------
        spec : ResampleSpec
            Resampling specification.

        Returns
        -------
        Self
            A **new** config instance with resampling parameters set.
        """
        return self._updated(resampling=spec)

    def _updated(self, **updates: object) -> Self:
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
