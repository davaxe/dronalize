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

from dataclasses import dataclass
from typing import TypedDict

import pandas as pd

from preprocessing.road_network.common import (
    CategoryStr,
    GraphBuilder,
    IntIDNode,
    MapGraph,
)
from preprocessing.road_network.edge_type import EdgeType
from preprocessing.road_network.waymo.protos import map_pb2, scenario_pb2


@dataclass
class WaymoScenario:
    """A Waymo scenario containing the scenario ID, map, and scenario data."""

    id: str
    map: WaymoMap
    scenario_data: pd.DataFrame

    @classmethod
    def from_proto(cls, scenario: scenario_pb2.Scenario) -> WaymoScenario:
        """Create a WaymoScenario from a Scenario proto."""
        return cls(
            id=scenario.scenario_id,
            map=WaymoMap.from_proto(list(scenario.map_features)),
            scenario_data=_scenario_data_from_proto(scenario),
        )

    @classmethod
    def from_data(cls, data: bytes) -> WaymoScenario:
        """Create a WaymoScenario from serialized data."""
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data)
        return cls.from_proto(scenario)

    def deconstruct(self) -> tuple[WaymoMap, pd.DataFrame]:
        """Deconstruct the WaymoScenario into its map and scenario data."""
        return self.map, self.scenario_data


class WaymoMap(GraphBuilder[int, IntIDNode]):
    """A Waymo map representation.

    As opposed to other maps (e.g, argoverse1, argoverse2 and NuScenes) this map
    parses the map features and builds a graph representation of the map.
    """

    def __init__(self) -> None:
        """Initialize the WaymoMap with empty features and nodes.

        The constructor `WaymoMap.from_proto` should be used to to create an
        instance from a list of `map_pb2.MapFeature` protos.
        """
        super().__init__()
        self.road_lines: dict[int, map_pb2.RoadLine] = {}
        self.road_edges: dict[int, map_pb2.RoadEdge] = {}
        self.driveways: dict[int, map_pb2.Driveway] = {}
        self.crosswalks: dict[int, map_pb2.Crosswalk] = {}
        self.lanes: dict[int, map_pb2.LaneCenter] = {}
        self.speed_bumps: dict[int, map_pb2.SpeedBump] = {}
        self.stop_signs: dict[int, map_pb2.StopSign] = {}

        self._no_map_features: bool = True

    @classmethod
    def from_proto(cls, map_features: list[map_pb2.MapFeature]) -> WaymoMap:
        """Create a WaymoMap from a list of MapFeature protos."""
        waymo_map = cls()
        waymo_map._process_map_features(map_features)
        waymo_map._no_map_features = len(map_features) == 0
        return waymo_map

    def empty_map(self) -> bool:
        """Check if the map is empty."""
        return self._no_map_features

    def new_node(self, x: float, y: float, z: float = 0) -> IntIDNode:
        """Create a new node with the given coordinates."""
        return IntIDNode(x=x, y=y, z=z)

    def build(
        self,
        *,
        interp_distance: float | None = None,
        min_distance: float | None = None,
    ) -> MapGraph:
        """Build the map graph from the Lyft LVL5 map.

        Args:
            interp_distance: the approximate distance between interpolated
                nodes. If None, no interpolation is performed.
            min_distance: the minimum distance between consecutive nodes. If `None`, no
                minimum distance is enforced.

        Returns:
            A `MapGraph` object representing the map graph.

        """
        self._processed_features: set[int] = set()
        self.min_distance = min_distance if min_distance is not None else 0.0

        self._add_road_edge_edges(interp_distance=interp_distance)
        self._add_road_line_edges(interp_distance=interp_distance)
        self._add_crosswalk_edges(interp_distance=interp_distance)
        self._add_driveway_edges(interp_distance=interp_distance)
        self._add_speed_bump_edges(interp_distance=interp_distance)
        self._add_lane_edges(interp_distance=interp_distance)
        self._add_stop_sign_nodes()

        return self.build_graph()

    def _process_map_features(
        self,
        map_features: list[map_pb2.MapFeature],
    ) -> None:
        """Process the map features and populate nodes and id_adj_list."""
        for feature in map_features:
            keys = [k.name for k, _ in feature.ListFields()]
            if "road_line" in keys:
                self.road_lines[feature.id] = feature.road_line
            elif "road_edge" in keys:
                self.road_edges[feature.id] = feature.road_edge
            elif "driveway" in keys:
                self.driveways[feature.id] = feature.driveway
            elif "crosswalk" in keys:
                self.crosswalks[feature.id] = feature.crosswalk
            elif "lane" in keys:
                self.lanes[feature.id] = feature.lane
            elif "speed_bump" in keys:
                self.speed_bumps[feature.id] = feature.speed_bump
            elif "stop_sign" in keys:
                self.stop_signs[feature.id] = feature.stop_sign

    def _add_speed_bump_edges(
        self,
        interp_distance: float | None = None,
    ) -> None:
        """Add edges for speed bumps to the map graph."""
        for feature_id, speed_bump in self.speed_bumps.items():
            if feature_id in self._processed_features:
                continue

            nodes = [
                self.new_node(x=point.x, y=point.y) for point in speed_bump.polygon
            ]
            self.add_node_edges_loop(
                nodes,
                is_polygon=True,
                interp_distance=interp_distance,
                edge_type=EdgeType.REGULATORY,
            )
            self._processed_features.add(feature_id)

    def _add_stop_sign_nodes(self) -> None:
        """Add stop sign nodes to the map graph."""
        for stop_sign in self.stop_signs.values():
            # Create a node for the stop sign location
            node = IntIDNode(x=stop_sign.position.x, y=stop_sign.position.y)
            self.add_node(node)

    def _add_lane_edges(
        self,
        *,
        interp_distance: float | None = None,
    ) -> None:
        """Process a lane center feature and update nodes and id_adj_list."""
        for feature_id, lane in self.lanes.items():
            if feature_id in self._processed_features:
                continue

            nodes = [IntIDNode(x=point.x, y=point.y) for point in lane.polyline]
            self.add_node_edges_loop_min_dist(
                nodes,
                min_distance=self.min_distance,
                is_polygon=False,
                interp_distance=interp_distance,
                edge_type=EdgeType.VIRTUAL,
            )
            self._processed_features.add(feature_id)

    def _add_crosswalk_edges(
        self,
        interp_distance: float | None = None,
    ) -> None:
        """Process a crosswalk feature and update nodes and id_adj_list."""
        for feature_id, crosswalk in self.crosswalks.items():
            if feature_id in self._processed_features:
                continue
            nodes = [IntIDNode(x=point.x, y=point.y) for point in crosswalk.polygon]

            self.add_node_edges_loop(
                nodes,
                is_polygon=True,
                interp_distance=interp_distance,
                edge_type=EdgeType.PEDESTRIAN_MARKING,
            )
            self._processed_features.add(feature_id)

    def _add_driveway_edges(
        self,
        interp_distance: float | None = None,
    ) -> None:
        """Process a driveway feature and update nodes and id_adj_list."""
        for feature_id, driveway in self.driveways.items():
            if feature_id in self._processed_features:
                continue
            nodes = [IntIDNode(x=point.x, y=point.y) for point in driveway.polygon]

            self.add_node_edges_loop_min_dist(
                nodes,
                min_distance=self.min_distance,
                is_polygon=True,
                interp_distance=interp_distance,
                edge_type=EdgeType.VIRTUAL,
            )
            self._processed_features.add(feature_id)

    def _add_road_edge_edges(
        self,
        interp_distance: float | None = None,
    ) -> None:
        """Process a road edge feature and update nodes and id_adj_list."""
        for feature_id, road_edge in self.road_edges.items():
            nodes = [IntIDNode(x=point.x, y=point.y) for point in road_edge.polyline]

            self.add_node_edges_loop_min_dist(
                nodes,
                min_distance=self.min_distance,
                is_polygon=False,
                interp_distance=interp_distance,
                edge_type=_ROAD_EDGE_TYPE_TO_EDGE_TYPE[road_edge.type],
            )
            self._processed_features.add(feature_id)

    def _add_road_line_edges(
        self,
        interp_distance: float | None = None,
    ) -> None:
        """Process a road line feature and update nodes and id_adj_list."""
        for feature_id, road_line in self.road_lines.items():
            nodes = [IntIDNode(x=point.x, y=point.y) for point in road_line.polyline]

            self.add_node_edges_loop_min_dist(
                nodes,
                min_distance=self.min_distance,
                is_polygon=False,
                interp_distance=interp_distance,
                edge_type=_ROAD_LINE_TYPE_TO_EDGE_TYPE[road_line.type],
            )
            self._processed_features.add(feature_id)


class _Frame(TypedDict):
    frame: int
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    psi: float
    category: CategoryStr
    scenario_id: str
    of_interest: bool


def _frame_from_object_state(
    object_state: scenario_pb2.ObjectState,
    frame: int,
    track_id: int,
    scene_id: str,
) -> _Frame | None:
    """Convert an ObjectState to a frame dictionary."""
    if not object_state.valid:
        return None

    return {
        "frame": frame,
        "track_id": track_id,
        "x": object_state.center_x,
        "y": object_state.center_y,
        "vx": object_state.velocity_x,
        "vy": object_state.velocity_y,
        "psi": object_state.heading,
        "category": "undefined",
        "scenario_id": scene_id,
        "of_interest": False,
    }


def _scenario_data_from_proto(
    scenario: scenario_pb2.Scenario,
) -> pd.DataFrame:
    """Convert a Scenario proto to a DataFrame."""
    data: list[_Frame] = []
    ego_vehicle_index = scenario.sdc_track_index
    objects_of_interest = list(scenario.objects_of_interest)

    for i, track in enumerate(scenario.tracks):
        track_id = track.id
        if i == ego_vehicle_index:
            track_id = -1  # Use -1 for the ego vehicle track ID

        for frame, state in enumerate(track.states):
            frame_data = _frame_from_object_state(
                state,
                frame=frame,
                track_id=track_id,
                scene_id=scenario.scenario_id,
            )
            if frame_data is None:
                continue

            frame_data["category"] = _OBJECT_TYPE_TO_CATEGORY.get(
                track.object_type,
                "undefined",
            )
            frame_data["of_interest"] = track.id in objects_of_interest
            data.append(frame_data)

    return pd.DataFrame(data)


_ROAD_LINE_TYPE_TO_EDGE_TYPE: dict[int, EdgeType] = {
    map_pb2.RoadLine.TYPE_UNKNOWN: EdgeType.VIRTUAL,
    map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: EdgeType.LINE_THIN_DASHED,
    map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: EdgeType.LINE_THIN_DOUBLE,
    map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: EdgeType.LINE_THIN,
    map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: EdgeType.LINE_THIN_DASHED,
    map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: EdgeType.LINE_THIN_DOUBLE_DASHED,
    map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: EdgeType.LINE_THIN,
    map_pb2.RoadLine.TYPE_SOLID_DOUBLE_YELLOW: EdgeType.LINE_THIN_DOUBLE,
    map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: EdgeType.LINE_THIN_DOUBLE,
}

# TODO: Double type???


_ROAD_EDGE_TYPE_TO_EDGE_TYPE: dict[int, EdgeType] = {
    map_pb2.RoadEdge.TYPE_UNKNOWN: EdgeType.VIRTUAL,
    map_pb2.RoadEdge.TYPE_ROAD_EDGE_BOUNDARY: EdgeType.ROAD_BORDER,
    map_pb2.RoadEdge.TYPE_ROAD_EDGE_MEDIAN: EdgeType.GUARD_RAIL,
}


_OBJECT_TYPE_TO_CATEGORY: dict[int, CategoryStr] = {
    scenario_pb2.Track.TYPE_UNSET: "undefined",
    scenario_pb2.Track.TYPE_VEHICLE: "car",
    scenario_pb2.Track.TYPE_PEDESTRIAN: "pedestrian",
    scenario_pb2.Track.TYPE_CYCLIST: "bicycle",
    scenario_pb2.Track.TYPE_OTHER: "undefined",
}
