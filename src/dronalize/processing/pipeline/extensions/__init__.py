"""Pipeline extensions for trajectory-processing assembly."""

from dronalize.processing.pipeline.extensions.base import TrajectoryPipelineExtension
from dronalize.processing.pipeline.extensions.lane_change import (
    LaneChangeSamplingExtension,
)

__all__ = [
    "LaneChangeSamplingExtension",
    "TrajectoryPipelineExtension",
]
