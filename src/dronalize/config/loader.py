from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dronalize.config.filtering import FilteringConfig
from dronalize.pipeline.functional.resample import Resampling, ResamplingMethod

if TYPE_CHECKING:
    from collections.abc import Collection

    from dronalize.categories import AgentCategory


def _normalize_frame_indices(
    frames: Collection[int],
    *,
    total_frames: int,
) -> frozenset[int]:
    """Normalize frame indices while preserving the original invalid input in errors."""
    normalized_frames: set[int] = set()
    for frame in frames:
        normalized_frame = frame if frame >= 0 else total_frames + frame
        if normalized_frame < 0 or normalized_frame >= total_frames:
            msg = f"Invalid frame index: {frame}"
            raise ValueError(msg)
        normalized_frames.add(normalized_frame)
    return frozenset(normalized_frames)


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

    input_len: int = Field(gt=0, description="Observation length in frames.")
    output_len: int = Field(gt=0, description="Prediction length in frames.")
    sample_time: float = Field(gt=0, description="Time interval between frames in seconds.")

    resampling: Resampling | None = Field(
        default=None, description="Resampling config if applicable."
    )
    window: WindowParams | None = Field(
        default=None,
        description=(
            "Used for datasets where multiple samples can be generated from a single "
            "scene by using a sliding window approach. If None, it is assumed that each "
            "scene corresponds to exactly one sample."
        ),
    )
    filtering: FilteringConfig | None = Field(
        default=None,
        description=(
            "Configuration for filtering scenes based on agent validity and scene composition."
        ),
    )
    extra_kwargs: dict[str, Any] = Field(
        default_factory=dict, description=("Extra keyword arguments to pass to the loader factory.")
    )

    @model_validator(mode="after")
    def _validate(self) -> LoaderConfig:
        if self.window is None:
            return self
        sequence_length = self._sequence_length()
        if self.window.window_size != sequence_length:
            msg = (
                f"Window size ({self.window.window_size}) must equal input_len + output_len "
                f"({sequence_length}) for consistent windowing."
            )
            raise ValueError(msg)
        return self

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
        return self.model_copy(update={"window": new_window_params})

    def with_filtering(
        self,
        min_agents: int = 2,
        *,
        require_all_valid: bool = False,
        require_frames: Collection[int] | None = None,
        filter_agent_category: Collection[AgentCategory] | None = None,
        filter_slow_agents: float | None = None,
        min_samples_per_agent: int | None = None,
    ) -> Self:
        """Return a copy with the given scene-filtering parameters.

        The inputs are passed into the `FilteringConfig` dataclass. Negative
        indices are supported in *require_frames*, where `-1` refers to the
        last frame of the sequence, `-2` to the second-to-last, etc.

        Parameters
        ----------
        min_agents : int, optional
            Minimum number of valid agents required in a scene. Defaults to 2.
        require_all_valid : bool, optional
            If True, all agents must have valid positions for every time step
            (input and output). Defaults to False.
        require_frames : Collection[int], optional
            Specific frame offsets (relative to scene start) that must be
            present. Supports negative indices. Defaults to None.
        filter_agent_category : Collection[AgentCategory], optional
            Agent categories to exclude from scenes. Defaults to None.
        filter_slow_agents : float, optional
            Remove agents whose average speed (m/s) is below this threshold.
            Defaults to None (no filtering).
        min_samples_per_agent : int, optional
            Minimum number of data points (rows) required per agent. Agents
            with fewer samples are removed. Defaults to None (no filtering).

        Returns
        -------
        Self
            A **new** config instance with scene-filtering parameters set.
        """
        normalized_frames = (
            _normalize_frame_indices(require_frames, total_frames=self._sequence_length())
            if require_frames is not None
            else None
        )

        new_filtering = FilteringConfig.create(
            min_agents=min_agents,
            require_all_valid=require_all_valid,
            require_frames=normalized_frames,
            filter_agent_category=filter_agent_category,
            filter_slow_agents=filter_slow_agents,
            min_samples_per_agent=min_samples_per_agent,
        )
        return self.model_copy(update={"filtering": new_filtering})

    def with_resampling(
        self,
        up: int,
        down: int,
        method: Literal["fast"] | ResamplingMethod = "fast",
    ) -> Self:
        """Return a copy with the given resampling parameters.

        Parameters
        ----------
        up : int
            Upsampling factor.
        down : int
            Downsampling factor.
        method : {"fast"}, optional
            Resampling method to use. Defaults to `"fast"`.

        Returns
        -------
        Self
            A **new** config instance with resampling parameters set.
        """
        new_resampling = Resampling(
            up=up,
            down=down,
            method=method if isinstance(method, ResamplingMethod) else ResamplingMethod(method),
        )
        return self.model_copy(update={"resampling": new_resampling})

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
