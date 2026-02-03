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

"""Graph builder abstraction for NuscenesMap."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, TypedDict
from uuid import UUID

from preprocessing.road_network.common import (
    GraphBuilder,
)
from preprocessing.road_network.edge_type import EdgeType
from preprocessing.road_network.nuscenes import parser

if TYPE_CHECKING:
    from pathlib import Path


class NuScenesMapGraphBuilder(GraphBuilder[str, parser.Node]):
    """A builder for creating a MapGraph from a NuscenesMap."""

    def __init__(
        self,
        nuscenes_map: parser.NuScenesMap,
        ignore_edge_types: set[str] | None = None,
    ) -> None:
        """Initialize the graph builder with a NuscenesMap."""
        super().__init__()
        self.map: parser.NuScenesMap = nuscenes_map
        self.map_nodes = self.map.nodes
        self.lane_polygon_edge: None | EdgeType = None
        self.edge_type_methods = {
            "road_divider": self._add_road_divider_edges,
            "lane_divider": self._add_lane_divider_edges,
            "walkway": self._add_walkway_edges,
            "pedestrian_crossing": self._add_pedestrian_crossing_edges,
            "traffic_light": self._add_traffic_light_edges,
            "stop_line": self._add_stop_line_edges,
            "lane": self._add_lane_edges,
            "carpark": self._add_carpark_edges,
        }
        self.min_distance: float = 0.0
        self.ignore_edge_types: set[str] = (
            ignore_edge_types if ignore_edge_types is not None else set()
        )

    @classmethod
    def from_json_file(
        cls,
        path: Path,
        *,
        debug_parsing: bool = False,
        ignore_edge_types: set[str] | None = None,
    ) -> Self:
        """Create a NuscenesMapGraphBuilder from a file path."""
        nuscenes_map = parser.NuScenesMap(
            path,
            enable_debug_prints=debug_parsing,
        )
        return cls(nuscenes_map, ignore_edge_types=ignore_edge_types)

    def new_node(self, x: float, y: float, z: float = 0) -> parser.Node:
        """Create a new node with the given coordinates."""
        return parser.Node(
            id=str(UUID(int=hash((x, y, z)) & ((1 << 128) - 1))),
            x=x,
            y=y,
            z=z,
        )

    def build_impl(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        """Build a `MapGraph` from the `NuscenesMap`.

        Args:
            interp_distance: the target distance for interpolation. If None,
                no interpolation is performed.
            min_distance: the minimum distance between consecutive nodes. If None,
                no minimum distance is enforced.

        """
        self.min_distance = min_distance if min_distance is not None else 0.0

        for edge_type, method in self.edge_type_methods.items():
            if edge_type not in self.ignore_edge_types:
                method(interp_distance=interp_distance)

    # --- Private methods ---

    def _add_road_divider_edges(self, interp_distance: float | None) -> None:
        for road_divider in self.map.road_dividers.values():
            line: parser.Line = self.map.lines[road_divider.line]
            self.add_node_edges_loop_min_dist(
                [self.map_nodes[i] for i in line.nodes],
                min_distance=self.min_distance,
                edge_type=EdgeType.LINE_THICK,
                interp_distance=interp_distance,
            )

    def _add_lane_divider_edges(self, interp_distance: float | None) -> None:
        for lane_divider in self.map.lane_dividers.values():
            self.add_node_edges_loop_min_dist(
                **self._extract_edges(lane_divider.segment_types),
                min_distance=self.min_distance,
                interp_distance=interp_distance,
            )

    def _add_lane_edges(self, interp_distance: float | None) -> None:
        for lane in self.map.lanes.values():
            if lane.left_lane_divider_segments:
                self.add_node_edges_loop_min_dist(
                    **self._extract_edges(lane.left_lane_divider_segments),
                    min_distance=self.min_distance,
                    interp_distance=interp_distance,
                )
            if lane.right_lane_divider_segments:
                self.add_node_edges_loop_min_dist(
                    **self._extract_edges(lane.right_lane_divider_segments),
                    min_distance=self.min_distance,
                    interp_distance=interp_distance,
                )

            if self.lane_polygon_edge is not None:
                lane_polygon: parser.Polygon = self.map.polygons[lane.polygon]
                nodes: list[str] = lane_polygon.exterior_nodes
                self.add_node_edges_loop_min_dist(
                    [self.map_nodes[i] for i in nodes],
                    min_distance=self.min_distance,
                    is_polygon=True,
                    edge_type=self.lane_polygon_edge,
                    interp_distance=interp_distance,
                )

    def _add_walkway_edges(self, interp_distance: float | None) -> None:
        for walkway in self.map.walkways.values():
            polygon: parser.Polygon = self.map.polygons[walkway.polygon]
            nodes: list[str] = polygon.exterior_nodes
            self.add_node_edges_loop(
                [self.map_nodes[i] for i in nodes],
                is_polygon=True,
                edge_type=EdgeType.CURB,
                interp_distance=interp_distance,
            )

    def _add_pedestrian_crossing_edges(self, interp_distance: float | None) -> None:
        for crossing in self.map.pedestrian_crossings.values():
            polygon: parser.Polygon = self.map.polygons[crossing.polygon]
            nodes: list[str] = polygon.exterior_nodes
            self.add_node_edges_loop(
                [self.map_nodes[i] for i in nodes],
                is_polygon=True,
                edge_type=EdgeType.PEDESTRIAN_MARKING,
                interp_distance=interp_distance,
            )

    def _add_traffic_light_edges(self, interp_distance: float | None) -> None:
        for traffic_light in self.map.traffic_lights.values():
            line: parser.Line = self.map.lines[traffic_light.line]
            self.add_node_edges_loop(
                [self.map_nodes[i] for i in line.nodes],
                edge_type=EdgeType.REGULATORY,
                interp_distance=interp_distance,
            )

    def _add_stop_line_edges(self, interp_distance: float | None) -> None:
        for stop_line in self.map.stop_lines.values():
            if not stop_line.is_valid(allow_pedestrian_crossings=False):
                # Skip turn stops as they are not represented as edges
                continue
            polygon: parser.Polygon = self.map.polygons[stop_line.polygon]
            nodes: list[str] = polygon.exterior_nodes
            self.add_node_edges_loop(
                [self.map_nodes[i] for i in nodes],
                is_polygon=True,
                edge_type=EdgeType.STOP,
                interp_distance=interp_distance,
            )

    def _add_carpark_edges(self, interp_distance: float | None) -> None:
        for carpark in self.map.carpark_areas.values():
            polygon: parser.Polygon = self.map.polygons[carpark.polygon]
            nodes: list[str] = polygon.exterior_nodes
            self.add_node_edges_loop(
                [self.map_nodes[i] for i in nodes],
                is_polygon=True,
                edge_type=EdgeType.VIRTUAL,
                interp_distance=interp_distance,
            )

    def _extract_edges(
        self,
        segments: list[tuple[str, parser.SegmentDividerType]],
    ) -> NuScenesMapGraphBuilder._Edges:
        return {
            "nodes": [self.map_nodes[n_id] for n_id, _ in segments],
            "edge_type": [
                parser.SegmentDividerType.to_edge_type(s_type)
                # The last segment type is always `NIL` as it is meaningless
                for _, s_type in segments[0:-1]
            ],
        }

    class _Edges(TypedDict):
        nodes: list[parser.Node]
        edge_type: list[EdgeType]
