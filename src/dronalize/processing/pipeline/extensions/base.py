"""Extension interfaces for trajectory pipeline assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from dronalize.config.models import ScenesConfig
    from dronalize.processing.pipeline.pipeline import Pipeline


class PipelineColumns(Protocol):
    """Minimal column mapping surface required by pipeline extensions."""

    @property
    def frame(self) -> str:
        """Return the frame column name."""
        ...

    @property
    def agent_id(self) -> str:
        """Return the agent-id column name."""
        ...

    @property
    def category(self) -> str | None:
        """Return the category column name, if any."""
        ...

    @property
    def lane_id(self) -> str | None:
        """Return the lane-id column name, if any."""
        ...


class PipelineBuildContext(Protocol):
    """Builder surface exposed to pipeline extensions."""

    @property
    def scenes(self) -> ScenesConfig | None:
        """Return the active scene-processing configuration."""
        ...

    @property
    def columns(self) -> PipelineColumns:
        """Return configured column names."""
        ...

    @property
    def scene_id_column(self) -> str:
        """Return the internal scene-id column name."""
        ...

    @property
    def has_window(self) -> bool:
        """Return whether the pipeline performs window extraction."""
        ...

    def window_group_columns(self) -> list[str]:
        """Return grouping columns used while generating windows."""
        ...

    def scene_key_columns(self) -> list[str]:
        """Return columns that identify one extracted scene."""
        ...

    def uses_scene_id(self) -> bool:
        """Return whether the pipeline uses an internal scene id."""
        ...

    def add_pre_window(self, pipeline: Pipeline) -> None:
        """Attach transforms before window extraction."""
        ...

    def add_post_window(self, pipeline: Pipeline) -> None:
        """Attach transforms after window extraction."""
        ...

    def add_post_screening(self, pipeline: Pipeline) -> None:
        """Attach transforms after scene screening."""
        ...


class TrajectoryPipelineExtension(Protocol):
    """Single entry point for extending trajectory pipeline assembly."""

    def extend(self, builder: PipelineBuildContext) -> None:
        """Mutate the given builder by attaching additional pipeline stages."""
        ...
