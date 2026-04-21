"""Execution-ready context shared by pipeline stage builders and extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.config.models import ScenesConfig
    from dronalize.processing.columns import TrajectoryColumns
    from dronalize.processing.models import PipelinePlan, SplitRequest


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Execution-ready context with all derived pipeline state precompiled."""

    plan: PipelinePlan
    columns: TrajectoryColumns
    window_by: tuple[str, ...]
    split_columns: tuple[str, ...]
    window_group_columns: tuple[str, ...]
    scene_key_columns: tuple[str, ...]
    scene_id_column: str | None
    drop_columns: tuple[str, ...]

    @property
    def frame_column(self) -> str:
        """Return the frame column configured by the trajectory spec."""
        return self.columns.frame

    @property
    def agent_id_column(self) -> str:
        """Return the agent-ID column configured by the trajectory spec."""
        return self.columns.agent_id

    @property
    def category_column(self) -> str:
        """Return the category column configured by the trajectory spec."""
        return self.columns.category

    @property
    def has_window(self) -> bool:
        """Return whether the pipeline will build sliding-window scenes."""
        return self.scenes.window is not None

    @property
    def split_request(self) -> SplitRequest | None:
        """Return the active split request, if the plan uses one."""
        return self.plan.split

    @property
    def scenes(self) -> ScenesConfig:
        """Return the scene-construction config from the pipeline plan."""
        return self.plan.scenes
