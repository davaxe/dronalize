from __future__ import annotations

from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Literal,
    Self,
)

from dronalize.common.trajectory.filter import FilteringConfig
from dronalize.common.trajectory.resample import Resampling

if TYPE_CHECKING:
    from collections.abc import Collection

    from dronalize.core.datatypes.categories import AgentCategory


@dataclass(slots=True, frozen=True)
class WindowParams:
    """Configuration for sliding window sampling of scenes."""

    window_size: int
    """Number of frames in each window."""

    step_size: int
    """Number of frames to skip between windows."""

    def __bool__(self) -> bool:
        """Return True if windowing is enabled (i.e., step_size > 0)."""
        return self.step_size > 0 and self.window_size > 0


@dataclass(slots=True, frozen=True)
class LoaderConfig:
    """Base configuration dataclass for trajectory data processing.

    All builder-style methods (`with_*`) return **new** instances, leaving the
    original unchanged.  This makes configs safe to share, compare, serialise
    and merge.
    """

    input_len: int
    """Observation length in frames."""

    output_len: int
    """Prediction length in frames."""

    sample_time: float
    """Time interval between frames in seconds."""

    resampling: Resampling | None = None
    """Resampling config if applicable."""

    window_params: WindowParams | None = None
    """Used for datasets where multiple samples can be generated from a single
    scene by using a sliding window approach. If None, it is assumed that each
    scene corresponds to exactly one sample."""

    scene_filtering: FilteringConfig | None = None
    """Configuration for filtering scenes based on agent validity and scene composition."""

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
        return replace(
            self,
            window_params=WindowParams(
                window_size=window_size
                if window_size is not None
                else self.input_len + self.output_len,
                step_size=step_size,
            ),
        )

    def with_filtering(
        self,
        min_agents: int = 2,
        *,
        require_all_valid: bool = False,
        require_frames: Collection[int] | None = None,
        filter_agent_category: Collection[AgentCategory] | None = None,
        filter_slow_agents: float | None = None,
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

        return replace(
            self,
            scene_filtering=FilteringConfig(
                min_agents=min_agents,
                require_all_valid=require_all_valid,
                require_frames=require_frames,
                filter_agent_category=filter_agent_category,
                filter_slow_agents=filter_slow_agents,
            ),
        )

    def with_resampling(
        self,
        up: int,
        down: int,
        method: Literal["fast", "spline"] = "fast",
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
        return replace(self, resampling=Resampling(up=up, down=down, method=method))
