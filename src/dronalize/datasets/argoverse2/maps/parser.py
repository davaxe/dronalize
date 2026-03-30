# Copyright 2024-2025, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dronalize.core.maps.edge_types import EdgeType

if TYPE_CHECKING:
    from dronalize.processing.maps.builder import Point


class Argoverse2Map:
    """Represents a map in the Argoverse2 dataset."""

    def __init__(self, json_file: Path) -> None:
        """Initialize the Argoverse2Map with a JSON file."""
        self.json_file: Path = json_file
        with Path.open(json_file, "r") as file:
            self.json_data: dict[str, Any] = json.load(file)

    @cached_property
    def segments(self) -> dict[int, LaneSegment]:
        """Get the lane segments from the JSON data."""
        segments_data: dict[str, dict[str, Any]] = self.json_data.get("lane_segments", [])
        return {segment["id"]: LaneSegment.from_dict(segment) for segment in segments_data.values()}

    @cached_property
    def pedestrian_crossings(self) -> dict[int, PedestrianCrossing]:
        """Get the pedestrian crossings from the JSON data."""
        crossings_data: dict[str, dict[str, Any]] = self.json_data.get("pedestrian_crossings", [])
        return {
            crossing["id"]: PedestrianCrossing.from_dict(crossing)
            for crossing in crossings_data.values()
        }

    @cached_property
    def drivable_areas(self) -> dict[int, DrivableArea]:
        """Get the drivable areas from the JSON data."""
        areas_data: dict[str, dict[str, Any]] = self.json_data.get("drivable_areas", [])
        return {area["id"]: DrivableArea.from_dict(area) for area in areas_data.values()}


class LaneType(IntEnum):
    """Represents the type of lane segment."""

    VEHICLE = 0
    BIKE = 1
    BUS = 2


class LaneBoundaryType(IntEnum):
    """Represents the type of lane marking."""

    NONE = auto()
    UNKNOWN = auto()
    SOLID_WHITE = auto()
    SOLID_YELLOW = auto()
    DOUBLE_SOLID_WHITE = auto()
    DOUBLE_SOLID_YELLOW = auto()
    DASHED_WHITE = auto()
    DASHED_YELLOW = auto()
    DOUBLE_DASH_WHITE = auto()
    DOUBLE_DASH_YELLOW = auto()
    DASH_SOLID_WHITE = auto()
    DASH_SOLID_YELLOW = auto()
    SOLID_DASH_WHITE = auto()
    SOLID_DASH_YELLOW = auto()
    SOLID_BLUE = auto()


@dataclass
class LaneBoundary:
    """Represents a lane boundary in the Argoverse2 map."""

    lane_type: LaneBoundaryType
    points: list[Point]

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, left: bool) -> LaneBoundary:
        """Create a `LaneBoundary` instance from a dictionary."""
        boundary_key = "left_lane_boundary" if left else "right_lane_boundary"
        type_key = "left_lane_mark_type" if left else "right_lane_mark_type"

        points: list[Point] = [(node["x"], node["y"]) for node in data[boundary_key]]
        return cls(lane_type=LaneBoundaryType[data[type_key]], points=points)

    def get_edge_type(self) -> EdgeType:
        """Get the corresponding `EdgeType` for the lane boundary."""
        return _BOUNDARY_TO_EDGE_TYPE.get(self.lane_type, EdgeType.NONE)


_BOUNDARY_TO_EDGE_TYPE = {
    # Seemingly NONE is often used and closest match is LINE_THIN.
    LaneBoundaryType.NONE: EdgeType.NONE,
    LaneBoundaryType.UNKNOWN: EdgeType.NONE,
    LaneBoundaryType.SOLID_WHITE: EdgeType.LINE_THIN,
    LaneBoundaryType.SOLID_YELLOW: EdgeType.LINE_THIN,
    LaneBoundaryType.DOUBLE_SOLID_WHITE: EdgeType.LINE_THIN_DOUBLE,
    LaneBoundaryType.DOUBLE_SOLID_YELLOW: EdgeType.LINE_THIN_DOUBLE,
    LaneBoundaryType.DASHED_WHITE: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.DASHED_YELLOW: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.DOUBLE_DASH_WHITE: EdgeType.LINE_THIN_DOUBLE_DASHED,
    LaneBoundaryType.DOUBLE_DASH_YELLOW: EdgeType.LINE_THIN_DOUBLE_DASHED,
    LaneBoundaryType.DASH_SOLID_WHITE: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.DASH_SOLID_YELLOW: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.SOLID_DASH_WHITE: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.SOLID_DASH_YELLOW: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.SOLID_BLUE: EdgeType.LINE_THICK,
}


@dataclass
class LaneSegment:
    """Represents a lane segment in the Argoverse2 map."""

    id: int
    lane_type: LaneType
    left_boundary: LaneBoundary | None = None
    right_boundary: LaneBoundary | None = None
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)
    right_neighbor_id: int | None = None
    left_neighbor_id: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LaneSegment:
        """Create a `LaneSegment` instance from a dictionary."""
        segment = cls(
            id=data["id"],
            lane_type=LaneType[data["lane_type"]],
            right_neighbor_id=data.get("right_neighbor_id"),
            left_neighbor_id=data.get("left_neighbor_id"),
        )

        if "left_lane_boundary" in data:
            segment.left_boundary = LaneBoundary.from_dict(data, left=True)
        if "right_lane_boundary" in data:
            segment.right_boundary = LaneBoundary.from_dict(data, left=False)

        segment.successors = data.get("successors", [])
        segment.predecessors = data.get("predecessors", [])

        return segment


@dataclass
class PedestrianCrossing:
    """Represents a pedestrian crossing in the Argoverse2 map."""

    id: int
    first_edge: list[Point]
    second_edge: list[Point]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PedestrianCrossing:
        """Create a `PedestrianCrossing` instance from a dictionary."""
        return cls(
            id=data["id"],
            first_edge=[(node["x"], node["y"]) for node in data["edge1"]],
            second_edge=[(node["x"], node["y"]) for node in data["edge2"]],
        )


@dataclass
class DrivableArea:
    """Represents a drivable area in the Argoverse2 map."""

    id: int
    boundary: list[Point] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DrivableArea:
        """Create a `DrivableArea` instance from a dictionary."""
        return cls(
            id=data["id"], boundary=[(node["x"], node["y"]) for node in data["area_boundary"]]
        )
