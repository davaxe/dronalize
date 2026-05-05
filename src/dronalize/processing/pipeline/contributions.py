"""Stage contribution types for trajectory pipeline extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from dronalize.processing.pipeline.pipeline import Pipeline


class PipelineStage(Enum):
    """Extension hook points in the trajectory pipeline."""

    PRE_WINDOW = auto()
    POST_WINDOW = auto()
    POST_SCREENING = auto()


@dataclass(frozen=True, slots=True)
class StageContributions:
    """Declarative stage-local contributions produced by one extension."""

    transforms: dict[PipelineStage, Pipeline] = field(default_factory=dict)
    require_scene_id: bool = False


def get_stage_pipelines(contributions: StageContributions, stage: PipelineStage) -> Pipeline:
    """Return the pipelines contributed to one stage."""
    return contributions.transforms.get(stage, Pipeline())
