"""Parser for NuScenes map data.

This module provides a structured interface for parsing and representing
map elements from the NuScenes dataset, such as nodes, lines, polygons,
lanes, road segments, dividers, stop lines, and other traffic-related objects.

It adheres to the NuScenes map schema (version 1.3) and converts JSON-based
map files into strongly typed Python objects for use in further processing and
converting to graph structures.

For details on the map structure, see:
https://www.nuscenes.org/nuscenes?tutorial=maps

NuScenes dataset documentation:
https://www.nuscenes.org/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import auto
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from typing_extensions import Self

from dronalize.core.datatypes.categories import EdgeType
from dronalize.core.datatypes.enum import BaseEnum

if TYPE_CHECKING:
    from collections.abc import Iterable


class NuScenesMap:
    """A class representing a NuScenes map, containing various map objects."""

    def __init__(
        self,
        json_file: Path,
    ) -> None:
        """Initialize the `NuscenesMap` with data from a JSON file.

        Parameters
        ----------
        json_file : Path
            Path to the JSON file containing the map data.

        """
        self.json_file: Path = json_file
        with Path.open(json_file) as f:
            self.json_data: dict[str, Any] = json.load(f)

    @cached_property
    def nodes(self) -> dict[str, Node]:
        """A dictionary of `Node` objects keyed by their str."""
        return _many_from_dict(Node, self.json_data["node"])

    @cached_property
    def lines(self) -> dict[str, Line]:
        """A dictionary of `Line` objects keyed by their str."""
        return _many_from_dict(Line, self.json_data["line"])

    @cached_property
    def polygons(self) -> dict[str, Polygon]:
        """A dictionary of `Polygon` objects keyed by their str."""
        return _many_from_dict(Polygon, self.json_data["polygon"])

    @cached_property
    def road_dividers(self) -> dict[str, RoadDivider]:
        """A dictionary of `RoadDivider` objects keyed by their str."""
        return _many_from_dict(
            RoadDivider,
            self.json_data["road_divider"],
        )

    @cached_property
    def road_segments(self) -> dict[str, RoadSegment]:
        """A dictionary of `RoadSegment` objects keyed by their str."""
        return _many_from_dict(
            RoadSegment,
            self.json_data["road_segment"],
        )

    @cached_property
    def pedestrian_crossings(self) -> dict[str, PedestrianCrossing]:
        """A dictionary of `PedestrianCrossing` objects keyed by their str."""
        return _many_from_dict(
            PedestrianCrossing,
            self.json_data["ped_crossing"],
        )

    @cached_property
    def walkways(self) -> dict[str, Walkway]:
        """A dictionary of `Walkway` objects keyed by their str."""
        return _many_from_dict(Walkway, self.json_data["walkway"])

    @cached_property
    def traffic_lights(self) -> dict[str, TrafficLight]:
        """A dictionary of `TrafficLight` objects keyed by their str."""
        return _many_from_dict(
            TrafficLight,
            self.json_data["traffic_light"],
        )

    @cached_property
    def lane_dividers(self) -> dict[str, LaneDivider]:
        """A dictionary of `LaneDivider` objects keyed by their str."""
        return _many_from_dict(
            LaneDivider,
            self.json_data["lane_divider"],
        )

    @cached_property
    def stop_lines(self) -> dict[str, StopLine]:
        """A dictionary of `StopLine` objects keyed by their str."""
        return _many_from_dict(
            StopLine,
            self.json_data["stop_line"],
        )

    @cached_property
    def lanes(self) -> dict[str, Lane]:
        """A dictionary of `Lane` objects keyed by their str."""
        return _many_from_dict(Lane, self.json_data["lane"])

    @cached_property
    def road_blocks(self) -> dict[str, RoadBlock]:
        """A dictionary of `RoadBlock` objects keyed by their str."""
        return _many_from_dict(
            RoadBlock,
            self.json_data["road_block"],
        )

    @cached_property
    def carpark_areas(self) -> dict[str, CarparkArea]:
        """A dictionary of `Carpark` objects keyed by their str."""
        return _many_from_dict(
            CarparkArea,
            self.json_data.get("carpark_area", []),
        )

    @cached_property
    def arcline_path_3(self) -> dict[str, ArclinePathV1]:
        """A dictionary of `ArclinePathV1` objects keyed by their str."""
        # This is dict[str, Any], where the key i the token and the value is the
        # arcline definition
        data: dict[str, Any] = self.json_data["arcline_path_3"]
        data_iter = (
            {"token": k, "knots": v[0], "ctrl": v[1], "order": v[2]} for k, v in data.items()
        )
        return _many_from_dict(ArclinePathV1, data_iter)


class StopLineType(BaseEnum):
    """Enum representing different types of stop lines in a NuScenes map."""

    TURN_STOP = auto()
    STOP_SIGN = auto()
    PEDESTRIAN_CROSSING = auto()
    TRAFFIC_LIGHT = auto()


class SegmentDividerType(BaseEnum):
    """Enum representing different types of segment dividers in a NuScenes map."""

    NIL = auto()
    DOUBLE_DASHED_WHITE = auto()
    SINGLE_SOLID_WHITE = auto()
    SINGLE_SOLID_YELLOW = auto()
    SINGLE_ZIGZAG_WHITE = auto()
    DOUBLE_SOLID_WHITE = auto()

    def to_edge_type(self) -> EdgeType:
        """Convert `SegmentDividerType` to dronalize `EdgeType`."""
        return _SEGMENT_DIVIDER_TYPE_TO_EDGE_TYPE.get(self, EdgeType.VIRTUAL)


_SEGMENT_DIVIDER_TYPE_TO_EDGE_TYPE: dict[SegmentDividerType, EdgeType] = {
    SegmentDividerType.NIL: EdgeType.VIRTUAL,
    SegmentDividerType.SINGLE_SOLID_WHITE: EdgeType.LINE_THIN,
    SegmentDividerType.SINGLE_SOLID_YELLOW: EdgeType.LINE_THIN,
    SegmentDividerType.SINGLE_ZIGZAG_WHITE: EdgeType.REGULATORY,
    SegmentDividerType.DOUBLE_SOLID_WHITE: EdgeType.LINE_THIN_DOUBLE,
    SegmentDividerType.DOUBLE_DASHED_WHITE: EdgeType.LINE_THIN_DOUBLE_DASHED,
}


class LaneType(BaseEnum):
    """Enum representing different types of lanes in a NuScenes map."""

    # First two are available in NuScenes. The rest are in View of Delft (VOD)
    NONE = auto()
    CAR = auto()

    # Need explicit value to easily convert from integers in VOD map file
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


class _FromDict(Protocol):
    """Protocol for classes that can be created from a dictionary."""

    id: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create an instance of the class from a dictionary."""
        ...


@dataclass
class Node:
    """A node in the NuScenes map, representing a point in space."""

    id: str
    x: float
    y: float

    def as_point(self) -> tuple[float, float]:
        """Return the node as an `(x, y)` point tuple."""
        return (self.x, self.y)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """Create a `Node` instance from a dictionary."""
        return Node(
            id=str(data["token"]),
            x=data["x"],
            y=data["y"],
        )


@dataclass
class ArclinePathV1:
    """A version 1 arcline path in the NuScenes map, representing a curved path."""

    id: str
    knots: list[float]
    ctrl: list[list[float]]
    order: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArclinePathV1:
        """Create a `ArclinePathV1` instance from a dictionary."""
        return ArclinePathV1(
            id=str(data["token"]),
            knots=data["knots"],
            ctrl=data["ctrl"],
            order=data["order"],
        )


@dataclass
class Line:
    """A line in the NuScenes map, representing a sequence of nodes."""

    id: str
    nodes: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Line:
        """Create a `Line` instance from a dictionary."""
        return Line(
            id=str(data["token"]),
            nodes=[str(node) for node in data["node_tokens"]],
        )


@dataclass
class Polygon:
    """A closed polygon in the NuScenes map, defined by nodes."""

    id: str
    exterior_nodes: list[str]
    holes: list[list[str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `Polygon` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            exterior_nodes=[str(node) for node in data["exterior_node_tokens"]],
            holes=[[str(node) for node in hole] for hole in data.get("interior_node_tokens", [])],
        )


@dataclass
class RoadDivider:
    """A road divider in NuScenes, represented as a line.

    Optionally, the corresponding road segment (as a reference to a
    `RoadSegment` type) can be specified.
    """

    id: str
    line: str
    road_segment: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `RoadDivider` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            line=str(data["line_token"]),
            road_segment=str(data.get("road_segment_token", "")) or None,
        )


@dataclass
class RoadSegment:
    """A road segment in NuScenes, defined by a polygon and a list of nodes."""

    id: str
    polygon: str
    is_intersection: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `RoadSegment` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
            is_intersection=data.get("is_intersection", False),
        )


@dataclass
class PedestrianCrossing:
    """A pedestrian crossing in NuScenes, represented by a polygon."""

    id: str
    polygon: str
    road_segment: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `PedestrianCrossing` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
            road_segment=str(data.get("road_segment_token", "")) or None,
        )


@dataclass
class Walkway:
    """A walkway in NuScenes, represented by a polygon."""

    id: str
    polygon: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `Walkway` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
        )


@dataclass
class TrafficLight:
    """A traffic light in NuScenes, represented by a line."""

    id: str
    line: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `TrafficLight` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            line=str(data["line_token"]),
        )


@dataclass
class LaneDivider:
    """A lane divider in NuScenes, represented by a line and segment types."""

    id: str
    line: str
    segment_types: list[tuple[str, SegmentDividerType]] = field(
        default_factory=list,
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `LaneDivider` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            line=str(data["line_token"]),
            segment_types=_parse_segment_divider(
                data.get("lane_divider_segments", []),
            ),
        )


@dataclass
class StopLine:
    """A stop line in NuScenes, represented by a polygon and associated objects."""

    id: str
    polygon: str
    stop_line_type: StopLineType
    pedestrian_crossings: list[str] = field(default_factory=list)
    traffic_lights: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `StopLine` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
            stop_line_type=StopLineType.from_str(
                data.get("stop_line_type", "TURN_STOP"),
            ),
            pedestrian_crossings=[str(pc) for pc in data.get("ped_crossing_tokens", [])],
            traffic_lights=[str(tl) for tl in data.get("traffic_light_tokens", [])],
        )

    def is_valid(self, *, allow_pedestrian_crossings: bool = False) -> bool:
        """Check if the stop line is valid for use in a map graph.

        Stops lines for pedestrian crossing can cause unwanted clutter in the
        graph, so they can be excluded by setting `allow_pedestrian_crossings`
        to `False`.
        """
        if not (self.traffic_lights or self.pedestrian_crossings):
            return False

        if self.stop_line_type == StopLineType.TURN_STOP:
            return False

        return not (
            self.stop_line_type == StopLineType.PEDESTRIAN_CROSSING
            and not allow_pedestrian_crossings
        )


@dataclass
class Lane:
    """A lane in NuScenes, represented by a polygon and lane dividers."""

    id: str
    polygon: str
    lane_type: LaneType = LaneType.NONE
    left_lane_divider_segments: list[tuple[str, SegmentDividerType]] = field(
        default_factory=list,
    )
    right_lane_divider_segments: list[tuple[str, SegmentDividerType]] = field(
        default_factory=list,
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `Lane` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
            lane_type=LaneType.try_from_str(data.get("lane_type", "NONE"))
            or LaneType.try_from_int(int(data.get("lane_type", -1)))
            or LaneType.NONE,
            left_lane_divider_segments=_parse_segment_divider(
                data.get("left_lane_divider_segments", []),
            ),
            right_lane_divider_segments=_parse_segment_divider(
                data.get("right_lane_divider_segments", []),
            ),
        )


@dataclass
class CarparkArea:
    """A carpark in NuScenes, represented by a polygon."""

    id: str
    polygon: str
    # Orientation of the parked cars in the carpark area in radians.
    orientation: float = 0.0
    road_block: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `CarparkArea` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
            orientation=data.get("orientation", 0.0),
            road_block=str(data.get("road_block_token", "")) or None,
        )


@dataclass
class RoadBlock:
    """A road block in NuScenes, represented by a polygon."""

    id: str
    polygon: str
    from_line_edge: str
    to_line_edge: str
    road_segment: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a `RoadBlock` instance from a dictionary."""
        return cls(
            id=str(data["token"]),
            polygon=str(data["polygon_token"]),
            from_line_edge=str(data["from_line_edge_token"]),
            to_line_edge=str(data["to_line_edge_token"]),
            road_segment=str(data["road_segment_token"]),
        )


def _parse_segment_divider(
    segments: list[dict[str, Any]],
) -> list[tuple[str, SegmentDividerType]]:
    """Parse a list of segment dividers from a dictionary.

    This is an internal utility method used to convert the
    dictionary representation of segment dividers into a list of tuples
    containing the node str and the segment divider type.
    """
    return [
        (
            str(segment_dict["node_token"]),
            SegmentDividerType.from_str(segment_dict["segment_type"]),
        )
        for segment_dict in segments
    ]


T = TypeVar("T", bound="_FromDict")


def _many_from_dict(
    cls: type[T],
    data: Iterable[dict[str, Any]],
) -> dict[str, T]:
    """Deserialize a sequence of dictionaries into a dictionary of objects.

    If an item cannot be deserialized, it is skipped. If `debug` is True, a
    warning is printed for each item that fails to deserialize.

    Parameters
    ----------
    cls : type[T]
        Class to deserialize the items into. Must have a ``from_dict``
        class method and an ``id`` attribute.
    data : Iterable[dict]
        An iterable of dictionaries representing the items to deserialize.

    Returns
    -------
    dict[str, T]
        A dictionary mapping UUIDs to deserialized objects of type ``cls``.

    """
    objects: dict[str, T] = {}
    for item in data:
        try:
            obj = cls.from_dict(item)
        except (ValueError, TypeError):
            continue
        objects[obj.id] = obj

    return objects
