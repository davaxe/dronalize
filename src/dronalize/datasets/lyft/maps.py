from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from dronalize.core.categories import EdgeType
from dronalize.datasets.lyft.protos import road_network_pb2 as proto
from dronalize.processing.maps import FeatureMapBuilder, PathFeature

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dronalize.processing.maps import Point


class LyftLVL5Map:
    """Represents a Lyft Level 5 map.

    The class provides access to various map elements such as lanes,
    road network segments, traffic control elements, road network nodes,
    and junctions.

    The access to these elements is provided through cached properties,
    which means that the elements are loaded only once and then cached
    for subsequent access.

    Parameters
    ----------
    protobuf_map_path : Path
        Path to the serialized protobuf map fragment.
    meta_json : Path
        Path to the metadata JSON containing the world transform.
    """

    def __init__(self, protobuf_map_path: Path, meta_json: Path) -> None:
        with Path.open(protobuf_map_path, "rb") as file:
            map_fragment = proto.MapFragment()
            _ = map_fragment.ParseFromString(file.read())

        with Path.open(meta_json, "r") as file:
            self.meta: dict[str, Any] = json.load(file)

        self._ecef_to_world: npt.NDArray[np.float64] = np.linalg.inv(
            np.array(self.meta["world_to_ecef"], dtype=np.float64)
        )
        self.elements: Sequence[proto.MapElement] = map_fragment.elements

    @cached_property
    def lanes(self) -> dict[str, Lane]:
        """Get all lanes in the map."""
        lanes: dict[str, Lane] = {}
        for el in self.elements:
            lane_id: str = _global_id_to_str(el.id)
            lane = Lane.from_proto_lane(
                lane_id, el.element.lane, transformation=self._ecef_to_world
            )
            lanes[_global_id_to_str(el.id)] = lane

        return lanes

    @cached_property
    def road_network_segments(self) -> dict[str, RoadNetworkSegment]:
        """Get all road network segments in the map."""
        segments: dict[str, RoadNetworkSegment] = {}
        for el in self.elements:
            if el.element.HasField("segment"):
                segment = RoadNetworkSegment.from_proto_road_network_segment(el.element.segment)
                segments[_global_id_to_str(el.id)] = segment

        return segments

    @cached_property
    def traffic_control_elements(self) -> dict[str, TrafficControlElement]:
        """Get all traffic control elements in the map."""
        elements: dict[str, TrafficControlElement] = {}
        for el in self.elements:
            if el.element.HasField("traffic_control_element"):
                element = TrafficControlElement.from_proto_traffic_control_element(
                    el.element.traffic_control_element, transformation=self._ecef_to_world
                )
                elements[_global_id_to_str(el.id)] = element

        return elements

    @cached_property
    def road_network_nodes(self) -> dict[str, RoadNetworkNode]:
        """Get all road network nodes in the map."""
        nodes: dict[str, RoadNetworkNode] = {}
        for el in self.elements:
            if el.element.HasField("node"):
                node = RoadNetworkNode.from_proto_road_network_node(el.element.node)
                nodes[_global_id_to_str(el.id)] = node

        return nodes

    @cached_property
    def junctions(self) -> dict[str, Junction]:
        """Get all junctions in the map."""
        junctions: dict[str, Junction] = {}
        for el in self.elements:
            if el.element.HasField("junction"):
                junction = Junction.from_proto_junction(el.element.junction)
                junctions[_global_id_to_str(el.id)] = junction

        return junctions


class LaneBoundaryType(IntEnum):
    """Types of lane boundaries (dividers) in the road network."""

    UNKNOWN = 0
    NONE = 1
    SINGLE_YELLOW_SOLID = 2
    SINGLE_WHITE_SOLID = 3
    SINGLE_YELLOW_DASHED = 4
    SINGLE_WHITE_DASHED = 5
    DOUBLE_YELLOW_SOLID = 6
    DOUBLE_WHITE_SOLID = 7
    DOUBLE_YELLOW_SOLID_FAR_DASHED_NEAR = 8
    DOUBLE_YELLOW_DASHED_FAR_SOLID_NEAR = 9
    CURB_RED = 10
    CURB_YELLOW = 11
    CURB = 12

    def to_edge_type(self) -> EdgeType:
        """Convert the lane boundary type to an edge type."""
        return _LANE_BOUNDARY_TYPE_TO_EDGE_TYPE[self]


_LANE_BOUNDARY_TYPE_TO_EDGE_TYPE = {
    LaneBoundaryType.UNKNOWN: EdgeType.VIRTUAL,
    LaneBoundaryType.NONE: EdgeType.VIRTUAL,
    LaneBoundaryType.SINGLE_YELLOW_SOLID: EdgeType.LINE_THIN,
    LaneBoundaryType.SINGLE_WHITE_SOLID: EdgeType.LINE_THIN,
    LaneBoundaryType.SINGLE_YELLOW_DASHED: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.SINGLE_WHITE_DASHED: EdgeType.LINE_THIN_DASHED,
    LaneBoundaryType.DOUBLE_YELLOW_SOLID: EdgeType.LINE_THIN_DOUBLE,
    LaneBoundaryType.DOUBLE_WHITE_SOLID: EdgeType.LINE_THIN_DOUBLE,
    LaneBoundaryType.DOUBLE_YELLOW_SOLID_FAR_DASHED_NEAR: EdgeType.LINE_THIN_DOUBLE_DASHED,
    LaneBoundaryType.DOUBLE_YELLOW_DASHED_FAR_SOLID_NEAR: EdgeType.LINE_THIN_DOUBLE_DASHED,
    LaneBoundaryType.CURB_RED: EdgeType.CURB,
    LaneBoundaryType.CURB_YELLOW: EdgeType.CURB,
    LaneBoundaryType.CURB: EdgeType.CURB,
}


class TurnType(IntEnum):
    """Types of turns in the lane segment."""

    UNKNOWN = 0
    THROUGH = 1
    LEFT = 2
    SHARP_LEFT = 3
    RIGHT = 4
    SHARP_RIGHT = 5
    U_TURN = 6


class RoadType(IntEnum):
    """Types of roads in the road network.

    Based on the OpenStreetMap road classification (from road_network.proto):
    https://wiki.openstreetmap.org/wiki/United_States_Road_Classification
    """

    UNKNOWN = 0
    MOTORWAY = 1
    TRUNK = 2
    PRIMARY = 3
    SECONDARY = 4
    TERTIARY = 5
    RESIDENTIAL = 6
    UNCLASSIFIED = 7
    SERVICE = 8
    SERVICE_PARKING_AISLE = 9
    SERVICE_DRIVEWAY = 10
    SERVICE_ALLEY = 11
    SERVICE_EMERGENCY_ACCESS = 12
    SERVICE_DRIVE_THROUGH = 13
    MOTORWAY_LINK = 14
    TRUNK_LINK = 15
    PRIMARY_LINK = 16
    SECONDARY_LINK = 17
    TERTIARY_LINK = 18
    SERVICE_LIVING_STREET = 19
    PEDESTRIAN = 20
    PATH = 21
    STEPS = 22
    CYCLEWAY = 23


class SideOfSegment(IntEnum):
    """Indicates the side of the road segment."""

    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2
    BOTH = 3
    NEITHER = 4


class TravelDirection(IntEnum):
    """Indicates the travel direction in the road network segment."""

    UNKNOWN = 0
    TWO_WAY = 1
    ONE_WAY_FORWARD = 2
    ONE_WAY_BACKWARD = 3
    ONE_WAY_REVERSIBLE = 4


@dataclass
class LaneSet:
    """Represents a set of (parallel) lanes in the road network."""

    num_driving_lanes: int
    """Total number of driving lanes (excluding DESIGNATED bike lanes)."""

    num_left_turn_lanes: int
    """Number of left turn lanes."""

    num_right_turn_lanes: int
    """Number of right turn lanes."""

    turn_description: str
    """Description of the lane set's turn, e.g., "left||"."""

    @classmethod
    def from_proto_lane_set(cls, lane_set: proto.RoadNetworkSegment.LaneSet) -> LaneSet:
        """Create a `LaneSet` instance from a `LaneSet` protobuf message."""
        return cls(
            num_driving_lanes=lane_set.num_driving_lanes,
            num_left_turn_lanes=lane_set.num_left_turn_driving_lanes,
            num_right_turn_lanes=lane_set.num_right_turn_driving_lanes,
            turn_description=lane_set.turn_descriptions_for_driving_lanes,
        )


@dataclass
class Junction:
    """Represents a junction in the road network."""

    is_non_trivial: bool
    """True for junction that corresponds to "real" intersections"""

    road_network_nodes: list[str]
    """Ids of the road network nodes connected to this junction."""

    extra_lanes: list[str]
    """Ids of the extra lanes that are not part of the road network segments."""

    traffic_control_elements: list[str]
    """Ids of the traffic control elements connected to this junction."""

    @classmethod
    def from_proto_junction(cls, junction: proto.Junction) -> Junction:
        """Create a `Junction` instance from a protobuf version."""
        return cls(
            is_non_trivial=junction.is_non_trivial_intersection,
            road_network_nodes=[_global_id_to_str(node) for node in junction.road_network_nodes],
            extra_lanes=[_global_id_to_str(lane) for lane in junction.lanes],
            traffic_control_elements=[
                _global_id_to_str(element) for element in junction.traffic_control_elements
            ],
        )


@dataclass
class RoadNetworkNode:
    """Represents a intersection node in the road network.

    Description of the node is taken from the protobuf definition:

    > "Map 'intersections'. There may be several nodes per junction, to
    represent sub-connections between different streets merging first etc."
    """

    latitude: float
    longitude: float
    altitude: float

    road_segments: list[str]
    """Ids of the road segments connected to this node."""

    junction: str | None = None
    """Id of the junction connected to this node, if any."""

    z_level: int = 0
    """Z-level of the node, used for elevation in 3D space."""

    @classmethod
    def from_proto_road_network_node(cls, node: proto.RoadNetworkNode) -> RoadNetworkNode:
        """Create a `RoadNetworkNode` instance from a protobuf version."""
        return cls(
            latitude=node.location.lat_e7 * 1e-7,
            longitude=node.location.lng_e7 * 1e-7,
            altitude=node.location.altitude_cm * 0.01,  # Convert cm to m
            road_segments=[_global_id_to_str(s) for s in node.road_segments],
            junction=_global_id_to_str(node.junction) if node.junction else None,
            z_level=node.z_level,
        )


@dataclass
class RoadNetworkSegment:
    """Represents a road segment in the road network.

    Essentially this is a collection of lanes that are connected.
    """

    start_node: str
    end_node: str
    forward_lane_set: LaneSet
    backward_lane_set: LaneSet
    num_bidirectional_lanes: int
    lanes: list[str]
    road_type: RoadType = RoadType.UNKNOWN
    travel_direction: TravelDirection = TravelDirection.UNKNOWN
    walkable: SideOfSegment = SideOfSegment.UNKNOWN

    @classmethod
    def from_proto_road_network_segment(
        cls, segment: proto.RoadNetworkSegment
    ) -> RoadNetworkSegment:
        """Create a `RoadNetworkSegment` instance from a protobuf version."""
        return cls(
            start_node=_global_id_to_str(segment.start_node),
            end_node=_global_id_to_str(segment.end_node),
            forward_lane_set=LaneSet.from_proto_lane_set(segment.forward_lane_set),
            backward_lane_set=LaneSet.from_proto_lane_set(segment.backward_lane_set),
            num_bidirectional_lanes=segment.num_bidirectional_lanes,
            lanes=[_global_id_to_str(lane) for lane in segment.lanes],
            road_type=RoadType(segment.road_class),
            travel_direction=TravelDirection(segment.travel_direction),
            walkable=SideOfSegment(segment.walkable),
        )

    def get_lane_id(self, lane_index: int) -> str:
        """Get the lane id and kind at the given index in the segment."""
        if lane_index < 0 or lane_index >= len(self.lanes):
            msg = (
                f"Lane index {lane_index} out of range for segment"
                f"{self.start_node} - {self.end_node}"
            )
            raise IndexError(msg)

        return self.lanes[lane_index]

    def lanes_iter(self) -> Iterable[str]:
        """Iterate over the lanes in the segment, yielding (lane_id, lane_kind)."""
        for i in range(len(self.lanes)):
            yield self.get_lane_id(i)


@dataclass
class LaneBoundary:
    """Represents a lane boundary in the road network."""

    lane_types: list[LaneBoundaryType]
    """One or more boundary types for te lane boundary."""

    points: list[Point]
    """List of points defining the lane boundary."""

    type_change_distances: list[float] | None = None
    """Distances in meters at which the lane boundary type changes. Expected to
    be one less than the number of lane types. If `None`, the lane boundary type
    does not change."""

    @classmethod
    def from_proto_lane_boundary(
        cls,
        boundary: proto.Lane.Boundary,
        frame: proto.GeoFrame,
        transformation: npt.NDArray[np.float64] | None = None,
    ) -> LaneBoundary:
        """Create a `LaneBoundary` instance from a `Lane` protobuf message."""
        cm_to_m: float = 0.01
        lane_types = [LaneBoundaryType(boundary_type) for boundary_type in boundary.divider_type]
        type_change_distances = [x * cm_to_m for x in iter(boundary.type_change_point_cm)]

        dx = [x * cm_to_m for x in boundary.vertex_deltas_x_cm]
        dy = [y * cm_to_m for y in boundary.vertex_deltas_y_cm]
        dz = [z * cm_to_m for z in boundary.vertex_deltas_z_cm]

        return cls(
            lane_types=lane_types,
            points=_parse_points(dx, dy, dz, frame, transformation),
            type_change_distances=type_change_distances or None,
        )

    def get_edge_type_from_src(self, src_idx: int) -> EdgeType:
        """Get the edge type for the lane boundary from a source index.

        This is the edge that start at the source index and goes to the next
        node in the lane boundary. If the source index is negative, it is
        interpreted as an index from the end of the list of nodes.

        Same results as using `get_edge_types()[src_idx]` if the source index
        is in the range of the number of nodes.
        """
        if src_idx < 0:
            src_idx = len(self.points) + src_idx

        if not self.lane_types:
            return EdgeType.VIRTUAL

        if not self.type_change_distances:
            return self.lane_types[0].to_edge_type()

        edge_types: list[EdgeType] = []
        acc_distance: float = 0.0
        change_count: int = 0
        edge_type: EdgeType = self.lane_types[0].to_edge_type()
        for i in range(len(self.points) - 1):
            edge_types.append(edge_type)
            dist = _point_distance(self.points[i], self.points[i + 1])
            acc_distance += dist

            if (
                change_count < len(self.type_change_distances)
                and acc_distance >= self.type_change_distances[change_count]
            ):
                change_count += 1
                edge_type = self.lane_types[change_count].to_edge_type()
                acc_distance = 0.0

            if i == src_idx:
                break

        return edge_type

    def get_edge_types(self) -> list[EdgeType] | EdgeType:
        """Get edge types for the lane boundary.

        Returns a list if the lane boundary changes with distance,
        otherwise returns a single edge type for the entire boundary.
        """
        if not self.lane_types:
            return EdgeType.NONE

        if not self.type_change_distances:
            return self.lane_types[0].to_edge_type()

        edge_types: list[EdgeType] = []
        acc_distance: float = 0.0
        change_count: int = 0
        edge_type: EdgeType = self.lane_types[0].to_edge_type()
        for i in range(len(self.points) - 1):
            edge_types.append(edge_type)
            dist = _point_distance(self.points[i], self.points[i + 1])
            acc_distance += dist

            if (
                change_count < len(self.type_change_distances)
                and acc_distance >= self.type_change_distances[change_count]
            ):
                change_count += 1
                edge_type = self.lane_types[change_count].to_edge_type()
                acc_distance = 0.0

        return edge_types


@dataclass
class Lane:
    """Represents a lane segment in the road network."""

    id: str
    parent_segment_or_junction: str

    left_boundary: LaneBoundary
    """Left boundary of the lane segment."""

    right_boundary: LaneBoundary
    """Right boundary of the lane segment."""

    turn_type: TurnType | None = None
    """Turn direction for lane segment if the parent segment is a junction."""

    travel_direction: TravelDirection = TravelDirection.UNKNOWN
    """Orientation in respect to the parent segment or junction."""

    lane_successors: list[str] = field(default_factory=list)
    """Ids of lane segments that are successors of this lane segment."""

    yield_to: list[str] = field(default_factory=list)
    """Ids of lane segments that this lane segment yields to."""

    can_have_parked_cars: bool = False
    """Indicates if the lane can have parked vehicles."""

    @classmethod
    def from_proto_lane(
        cls, lane_id: str, lane: proto.Lane, transformation: npt.NDArray[np.float64] | None = None
    ) -> Lane:
        """Create a `Lane` instance from a protobuf message."""
        return cls(
            id=lane_id,
            parent_segment_or_junction=_global_id_to_str(lane.parent_segment_or_junction),
            left_boundary=LaneBoundary.from_proto_lane_boundary(
                lane.left_boundary, lane.geo_frame, transformation
            ),
            right_boundary=LaneBoundary.from_proto_lane_boundary(
                lane.right_boundary, lane.geo_frame, transformation
            ),
            travel_direction=TravelDirection(lane.orientation_in_parent_segment),
            lane_successors=[_global_id_to_str(successor) for successor in lane.lanes_ahead],
            can_have_parked_cars=lane.can_have_parked_cars,
            turn_type=TurnType(lane.turn_type_in_parent_junction),
        )


class TrafficControlElementType(IntEnum):
    """Types of traffic control elements."""

    NOT_RELEVANT = auto()  # Not relevant in the context of this project
    TRAFFIC_LIGHT = auto()
    YIELD_SIGN = auto()
    STOP_SIGN = auto()
    STOP_LINE = auto()
    SPEED_BUMP = auto()
    SPEED_HUMP = auto()
    PEDESTRIAN_CROSSING = auto()
    KEEP_CLEAR = auto()
    CONSTRUCTION_ZONE = auto()
    STOP_FOR_PEDESTRIANS = auto()


class GeometryType(IntEnum):
    """Types of geometry for traffic control elements."""

    UNKNOWN = 0
    POINT = 1
    MULTI_POINT = 2
    LINESTRING = 3
    POLYGON = 4


@dataclass
class TrafficControlElement:
    """Represents a traffic control element in the road network."""

    control_element_type: TrafficControlElementType
    """Type of the traffic control element, e.g., TRAFFIC_LIGHT, STOP_SIGN, etc."""

    geometry_type: GeometryType
    """Type of geometry for the traffic control element"""

    points: list[Point] | None = None
    """Points defining the geometry of this traffic control element."""

    controlled_lane_sequence: list[str] = field(default_factory=list)
    """Ids of the lanes controlled by this traffic control element.

    The first lane in the sequence is the lane affected by the traffic control
    element (where it is observed/located), and the subsequent lanes are
    lanes in which you can proceed after the traffic control element.
    """

    @classmethod
    def from_proto_traffic_control_element(
        cls,
        element: proto.TrafficControlElement,
        transformation: npt.NDArray[np.float64] | None = None,
    ) -> TrafficControlElement:
        """Create a `TrafficControlElement` instance from a protobuf message."""
        cm_to_m = 0.01
        dx = [x * cm_to_m for x in element.points_x_deltas_cm]
        dy = [y * cm_to_m for y in element.points_y_deltas_cm]
        dz = [z * cm_to_m for z in element.points_z_deltas_cm]

        if not dx or not dy or not dz:
            points = None
        else:
            points = _parse_points(dx, dy, dz, element.geo_frame, transformation)

        return cls(
            control_element_type=cls._solve_type(element),
            geometry_type=GeometryType(int(element.geometry_type)),
            points=points,
        )

    @staticmethod
    def _solve_type(element: proto.TrafficControlElement) -> TrafficControlElementType:
        """Solve the type of the traffic control element."""
        keys = {key.name for key, _ in element.ListFields()}
        key_to_type = {
            "traffic_light": TrafficControlElementType.TRAFFIC_LIGHT,
            "yield_sign": TrafficControlElementType.YIELD_SIGN,
            "stop_sign": TrafficControlElementType.STOP_SIGN,
            "stop_line": TrafficControlElementType.STOP_LINE,
            "speed_bump": TrafficControlElementType.SPEED_BUMP,
            "speed_hump": TrafficControlElementType.SPEED_HUMP,
            "pedestrian_crosswalk": TrafficControlElementType.PEDESTRIAN_CROSSING,
            "keep_clear": TrafficControlElementType.KEEP_CLEAR,
            "construction_zone": TrafficControlElementType.CONSTRUCTION_ZONE,
            "stop_for_pedestrians": TrafficControlElementType.STOP_FOR_PEDESTRIANS,
        }
        for key, tce_type in key_to_type.items():
            if key in keys:
                return tce_type
        return TrafficControlElementType.NOT_RELEVANT


def _global_id_to_str(global_id: proto.GlobalId) -> str:
    """Convert a GlobalId to a string."""
    return global_id.id.decode("utf-8") if global_id.id else ""


def _parse_points(
    dx: Sequence[float],
    dy: Sequence[float],
    dz: Sequence[float],
    frame: proto.GeoFrame,
    transformation: npt.NDArray[np.float64] | None = None,
) -> list[Point]:
    if not dx or not dy or not dz:
        return []

    lat, lon = frame.origin.lat_e7 * 1e-7, frame.origin.lng_e7 * 1e-7
    lat, lon = math.radians(lat), math.radians(lon)
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    z = np.cumsum(dz)

    result: list[Point] = []
    for xi, yi, zi in zip(x, y, z, strict=True):
        tx, ty, _tz = transform(xi, yi, zi, lat, lon, transformation)
        result.append((tx, ty))
    return result


def _point_distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def transform(
    e: float,
    n: float,
    u: float,
    lat: float,
    lon: float,
    transformation: npt.NDArray[np.float64] | None = None,
) -> tuple[float, float, float]:
    """Transform lyft map nodes to world coordinates."""
    x, y, z = _enu_to_ecef(e, n, u, lat, lon)
    if transformation is None:
        return (x, y, z)

    transformation = np.transpose(transformation)
    xyz = (np.array([x, y, z]) @ transformation[:3, :3] + transformation[-1:, :3]).flatten()
    return (xyz[0], xyz[1], xyz[2])


def _enu_to_ecef(
    e: float, n: float, u: float, lat: float, lon: float
) -> tuple[float, float, float]:
    x0, y0, z0 = _geodetic_to_ecef(lat, lon, 0.0)
    dx, dy, dz = _enu_to_uvw(e, n, u, lat, lon)
    return x0 + dx, y0 + dy, z0 + dz


def _geodetic_to_ecef(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
    # WGS84 constants
    semi_major_axis = 6378137.0  # meters
    semi_minor_axis = 6356752.31424518

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Radius of curvature in the prime vertical
    n = semi_major_axis**2 / math.hypot(semi_major_axis * cos_lat, semi_minor_axis * sin_lat)

    # ECEF coordinates
    x = (n + alt) * cos_lat * cos_lon
    y = (n + alt) * cos_lat * sin_lon
    z = (n * (semi_minor_axis / semi_major_axis) ** 2 + alt) * sin_lat

    return x, y, z


def _enu_to_uvw(
    east: float, north: float, up: float, lat: float, lon: float
) -> tuple[float, float, float]:
    cos_lat = math.cos(lat)
    sin_lat = math.sin(lat)
    cos_lon = math.cos(lon)
    sin_lon = math.sin(lon)
    t = cos_lat * up - sin_lat * north
    w = sin_lat * up + cos_lat * north
    u = cos_lon * t - sin_lon * east
    v = sin_lon * t + cos_lon * east
    return u, v, w


class LyftMapBuilder(FeatureMapBuilder):
    """Builder for a map graph from a Lyft LVL5 map."""

    def __init__(self, lyft_map: LyftLVL5Map) -> None:
        self.map: LyftLVL5Map = lyft_map

    @classmethod
    def from_files(cls, map_path: Path, meta_json: Path) -> LyftMapBuilder:
        """Create a map builder from map and metadata files."""
        return cls(LyftLVL5Map(map_path, meta_json))

    @override
    def iter_features(self) -> Iterable[PathFeature]:
        added_lanes: set[str] = set()
        for road_segment in self.map.road_network_segments.values():
            for lane_id in road_segment.lanes_iter():
                if lane_id in added_lanes:
                    continue
                yield from self._lane_features(self.map.lanes[lane_id], junction=False)
                added_lanes.add(lane_id)

        for junction in self.map.junctions.values():
            for lane_id in junction.extra_lanes:
                if lane_id in added_lanes:
                    continue
                yield from self._lane_features(self.map.lanes[lane_id], junction=True)
                added_lanes.add(lane_id)

    @staticmethod
    def _lane_features(lane: Lane, *, junction: bool) -> Iterable[PathFeature]:
        boundaries = (
            (lane.left_boundary,) if junction else (lane.left_boundary, lane.right_boundary)
        )
        for boundary in boundaries:
            yield PathFeature(
                points=tuple(boundary.points),
                edge_types=tuple(
                    boundary.get_edge_type_from_src(i) for i in range(len(boundary.points) - 1)
                ),
            )
