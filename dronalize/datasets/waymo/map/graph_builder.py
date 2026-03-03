from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.core.datatypes.categories import EdgeType
from dronalize.core.graph import GraphBuilder, IntIDNode
from dronalize.datasets.waymo.protos import lean_map_pb2

if TYPE_CHECKING:
    from collections.abc import Sequence


class WaymoMapGraphBuilder(GraphBuilder[int, IntIDNode]):
    """A Waymo map representation.

    As opposed to other maps (e.g, argoverse1, argoverse2 and NuScenes) this map
    parses the map features and builds a graph representation of the map.
    """

    def __init__(self) -> None:
        """Initialize the WaymoMapGraphBuilder with empty features and nodes.

        The constructor `WaymoMapGraphBuilder.from_proto` should be used to to
        create an instance from a list of `lean_map_pb2.MapFeature` protos.
        """
        super().__init__()
        self.road_lines: dict[int, lean_map_pb2.RoadLine] = {}
        self.road_edges: dict[int, lean_map_pb2.RoadEdge] = {}
        self.driveways: dict[int, lean_map_pb2.Driveway] = {}
        self.crosswalks: dict[int, lean_map_pb2.Crosswalk] = {}
        self.lanes: dict[int, lean_map_pb2.LaneCenter] = {}
        self.speed_bumps: dict[int, lean_map_pb2.SpeedBump] = {}
        self.stop_signs: dict[int, lean_map_pb2.StopSign] = {}

        self._no_map_features: bool = True

    @classmethod
    def from_proto(cls, map_features: Sequence[lean_map_pb2.MapFeature]) -> WaymoMapGraphBuilder:
        """Create a WaymoMapGraphBuilder from a list of MapFeature protos."""
        waymo_map = cls()
        waymo_map._process_map_features(map_features)
        waymo_map._no_map_features = len(map_features) == 0
        return waymo_map

    def empty_map(self) -> bool:
        """Check if the map is empty."""
        return self._no_map_features

    @override
    def new_node(self, x: float, y: float, z: float = 0) -> IntIDNode:
        return IntIDNode(self.next_node_id(), x=x, y=y, z=z)

    @override
    def build_impl(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        self._processed_features: set[int] = set()

        self._add_road_edge_edges(min_distance, interp_distance)
        self._add_road_line_edges(min_distance, interp_distance)
        self._add_crosswalk_edges(min_distance, interp_distance)
        self._add_driveway_edges(min_distance, interp_distance)
        self._add_speed_bump_edges(min_distance, interp_distance)
        self._add_lane_edges(min_distance, interp_distance)
        self._add_stop_sign_nodes()

    def _process_map_features(
        self,
        map_features: Sequence[lean_map_pb2.MapFeature],
    ) -> None:
        """Process the map features and populate nodes and id_adj_list."""
        r_lines = self.road_lines
        r_edges = self.road_edges
        dw = self.driveways
        cw = self.crosswalks
        ln = self.lanes
        sb = self.speed_bumps
        ss = self.stop_signs

        for feature in map_features:
            kind = feature.WhichOneof("feature_data")
            if kind == "road_line":
                r_lines[feature.id] = feature.road_line
            elif kind == "road_edge":
                r_edges[feature.id] = feature.road_edge
            elif kind == "lane":
                ln[feature.id] = feature.lane
            elif kind == "stop_sign":
                ss[feature.id] = feature.stop_sign
            elif kind == "crosswalk":
                cw[feature.id] = feature.crosswalk
            elif kind == "speed_bump":
                sb[feature.id] = feature.speed_bump
            elif kind == "driveway":
                dw[feature.id] = feature.driveway

    def _add_speed_bump_edges(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        """Add edges for speed bumps to the map graph."""
        for feature_id, speed_bump in self.speed_bumps.items():
            if feature_id in self._processed_features:
                continue

            nodes = [self.new_node(x=point.x, y=point.y) for point in speed_bump.polygon]
            self.add_node_edges_loop_min_dist(
                nodes,
                is_polygon=True,
                min_distance=min_distance,
                interp_distance=interp_distance,
                edge_type=EdgeType.REGULATORY,
            )
            self._processed_features.add(feature_id)

    def _add_stop_sign_nodes(self) -> None:
        """Add stop sign nodes to the map graph."""
        for stop_sign in self.stop_signs.values():
            # Create a node for the stop sign location
            node = self.new_node(x=stop_sign.position.x, y=stop_sign.position.y)
            self.add_node(node)

    def _add_lane_edges(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        """Process a lane center feature and update nodes and id_adj_list."""
        for feature_id, lane in self.lanes.items():
            if feature_id in self._processed_features:
                continue

            nodes = [self.new_node(x=point.x, y=point.y) for point in lane.polyline]
            self.add_node_edges_loop_min_dist(
                nodes,
                is_polygon=False,
                min_distance=min_distance,
                interp_distance=interp_distance,
                edge_type=EdgeType.VIRTUAL,
            )
            self._processed_features.add(feature_id)

    def _add_crosswalk_edges(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        """Process a crosswalk feature and update nodes and id_adj_list."""
        for feature_id, crosswalk in self.crosswalks.items():
            if feature_id in self._processed_features:
                continue
            nodes = [self.new_node(x=point.x, y=point.y) for point in crosswalk.polygon]

            self.add_node_edges_loop_min_dist(
                nodes,
                is_polygon=True,
                min_distance=min_distance,
                interp_distance=interp_distance,
                edge_type=EdgeType.PEDESTRIAN_MARKING,
            )
            self._processed_features.add(feature_id)

    def _add_driveway_edges(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        """Process a driveway feature and update nodes and id_adj_list."""
        for feature_id, driveway in self.driveways.items():
            if feature_id in self._processed_features:
                continue
            nodes = [self.new_node(x=point.x, y=point.y) for point in driveway.polygon]

            self.add_node_edges_loop_min_dist(
                nodes,
                is_polygon=True,
                min_distance=min_distance,
                interp_distance=interp_distance,
                edge_type=EdgeType.VIRTUAL,
            )
            self._processed_features.add(feature_id)

    def _add_road_edge_edges(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        """Process a road edge feature and update nodes and id_adj_list."""
        for feature_id, road_edge in self.road_edges.items():
            nodes = [self.new_node(x=point.x, y=point.y) for point in road_edge.polyline]

            self.add_node_edges_loop_min_dist(
                nodes,
                is_polygon=False,
                min_distance=min_distance,
                interp_distance=interp_distance,
                edge_type=_ROAD_EDGE_TYPE_TO_EDGE_TYPE[road_edge.type],
            )
            self._processed_features.add(feature_id)

    def _add_road_line_edges(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        """Process a road line feature and update nodes and id_adj_list."""
        for feature_id, road_line in self.road_lines.items():
            nodes = [self.new_node(x=point.x, y=point.y) for point in road_line.polyline]

            self.add_node_edges_loop_min_dist(
                nodes,
                min_distance=min_distance,
                is_polygon=False,
                interp_distance=interp_distance,
                edge_type=_ROAD_LINE_TYPE_TO_EDGE_TYPE[road_line.type],
            )
            self._processed_features.add(feature_id)


_ROAD_LINE_TYPE_TO_EDGE_TYPE: dict[int, EdgeType] = {
    lean_map_pb2.RoadLine.TYPE_UNKNOWN: EdgeType.VIRTUAL,
    lean_map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: EdgeType.LINE_THIN_DASHED,
    lean_map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: EdgeType.LINE_THIN_DOUBLE,
    lean_map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: EdgeType.LINE_THIN,
    lean_map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: EdgeType.LINE_THIN_DASHED,
    lean_map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: EdgeType.LINE_THIN_DOUBLE_DASHED,
    lean_map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: EdgeType.LINE_THIN,
    lean_map_pb2.RoadLine.TYPE_SOLID_DOUBLE_YELLOW: EdgeType.LINE_THIN_DOUBLE,
    lean_map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: EdgeType.LINE_THIN_DOUBLE,
}

_ROAD_EDGE_TYPE_TO_EDGE_TYPE: dict[int, EdgeType] = {
    lean_map_pb2.RoadEdge.TYPE_UNKNOWN: EdgeType.VIRTUAL,
    lean_map_pb2.RoadEdge.TYPE_ROAD_EDGE_BOUNDARY: EdgeType.ROAD_BORDER,
    lean_map_pb2.RoadEdge.TYPE_ROAD_EDGE_MEDIAN: EdgeType.GUARD_RAIL,
}
