from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dronalize.ops.trajectory.filter import FilteringConfig
from dronalize.ops.trajectory.resample import Resampling, ResamplingMethod

if TYPE_CHECKING:
    from collections.abc import Collection

    from dronalize.core.datatypes.categories import AgentCategory


class WindowParams(BaseModel):
    """Configuration for sliding window sampling of scenes."""

    model_config = ConfigDict(frozen=True)
    window_size: int = Field(gt=0, description="Number of frames in each window.")
    step_size: int = Field(gt=0, description="Number of frames to skip between windows.")

    def __bool__(self) -> bool:
        """Return True if windowing is enabled (i.e., step_size > 0)."""
        return self.step_size > 0 and self.window_size > 0


class LoaderConfig(BaseModel):
    """Base configuration class for trajectory data processing.

    All builder-style methods (`with_*`) return **new** instances, leaving the
    original unchanged. This makes configs safe to share, compare, serialise
    and merge.
    """

    model_config = ConfigDict(frozen=True)

    input_len: int = Field(gt=0, description="Observation length in frames.")
    output_len: int = Field(gt=0, description="Prediction length in frames.")
    sample_time: float = Field(gt=0, description="Time interval between frames in seconds.")

    resampling: Resampling | None = Field(
        default=None, description="Resampling config if applicable."
    )
    window: WindowParams | None = Field(
        default=None,
        description="Used for datasets where multiple samples can be generated from a single "
        "scene by using a sliding window approach. If None, it is assumed that each "
        "scene corresponds to exactly one sample.",
    )
    filtering: FilteringConfig | None = Field(
        default=None,
        description=(
            "Configuration for filtering scenes based on agent validity and scene composition."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> LoaderConfig:
        if self.window is None:
            return self
        if self.window.window_size != self.input_len + self.output_len:
            msg = (
                f"Window size ({self.window.window_size}) must equal input_len + output_len "
                f"({self.input_len + self.output_len}) for consistent windowing."
            )
            raise ValueError(msg)
        return self

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
            window_size=window_size
            if window_size is not None
            else self.input_len + self.output_len,
            step_size=step_size,
        )
        # Replaces dataclasses.replace()
        return self.model_copy(update={"window_params": new_window_params})

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
        if require_frames is not None:
            require_frames = {
                frame if frame > 0 else (self.input_len + self.output_len + frame)
                for frame in require_frames
            }

        new_filtering = FilteringConfig(
            min_agents=min_agents,
            require_all_valid=require_all_valid,
            require_frames=frozenset(require_frames) if require_frames is not None else None,
            filter_agent_category=frozenset(filter_agent_category)
            if filter_agent_category is not None
            else None,
            filter_slow_agents=filter_slow_agents,
            min_samples_per_agent=min_samples_per_agent,
        )
        return self.model_copy(update={"scene_filtering": new_filtering})

    def with_resampling(
        self,
        up: int,
        down: int,
        method: Literal["fast", "spline"] | ResamplingMethod = "fast",
    ) -> Self:
        """Return a copy with the given resampling parameters.

        Parameters
        ----------
        up : int
            Upsampling factor.
        down : int
            Downsampling factor.
        method : {"fast", "spline"}, optional
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
