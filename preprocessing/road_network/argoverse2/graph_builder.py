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

from typing import TYPE_CHECKING

from preprocessing.road_network.argoverse2 import parser
from preprocessing.road_network.common import (
    GraphBuilder,
    IntIDNode,
    MapGraph,
)
from preprocessing.road_network.edge_type import EdgeType

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class Argoverse2GraphBuilder(GraphBuilder[int, IntIDNode]):
    """A builder for creating a graph representation of an Argoverse2 map."""

    def __init__(self, map_data: parser.Argoverse2Map) -> None:
        """Initialize the graph builder with an Argoverse2Map."""
        super().__init__()
        self.map = map_data

        self.assume_none_as: EdgeType = EdgeType.VIRTUAL

    @classmethod
    def from_json_file(cls, json_file: Path) -> Argoverse2GraphBuilder:
        """Create an `Argoverse2GraphBuilder` from a JSON file."""
        map_data = parser.Argoverse2Map(json_file)
        return cls(map_data)

    def new_node(self, x: float, y: float, z: float = 0) -> IntIDNode:
        """Create a new node with the given coordinates."""
        return IntIDNode(x, y, z)

    def build(
        self,
        *,
        interp_distance: float | None = None,
    ) -> MapGraph:
        """Build a `MapGraph` from the `Argoverse2Map`.

        To perform interpolation, set `interpolate` to True and provide a value
        for `interp_distance`.

        Args:
            interp_distance: the target distance for interpolation. If None,
                no interpolation is performed.

        Returns:
            A `MapGraph` object containing the node positions, edge indices, and
            edge types.

        """
        self._add_lane_boundary_edges(
            interp_distance=interp_distance,
        )
        self._add_pedestrian_crossing_edges(
            interp_distance=interp_distance,
        )

        return self.build_graph()

    # --- Private methods ----

    def _add_pedestrian_crossing_edges(
        self,
        *,
        interp_distance: float | None = None,
    ) -> None:
        """Add edges for pedestrian crossings in the map."""
        for crossing in self.map.pedestrian_crossings.values():
            self.add_node_edges_loop(
                nodes=crossing.first_edge,
                interp_distance=interp_distance,
                edge_type=EdgeType.PEDESTRIAN_MARKING,
            )
            self.add_node_edges_loop(
                nodes=crossing.second_edge,
                interp_distance=interp_distance,
                edge_type=EdgeType.PEDESTRIAN_MARKING,
            )

    def _add_lane_boundary_edges(
        self,
        *,
        interp_distance: float | None = None,
    ) -> None:
        """Add edges for lane boundaries in the map."""
        lane_segments = self.map.segments.copy()

        while lane_segments:
            segment_id, segment = lane_segments.popitem()
            self.add_edges_from_iterable(
                self._traverse_lane_segment(
                    segment_id,
                    interp_distance=interp_distance,
                ),
            )

    def _traverse_lane_segment(
        self,
        start: int,
        *,
        interp_distance: float | None = None,
    ) -> Iterable[tuple[int, int, EdgeType]]:
        """Traverse a lane segment and yield edges for its boundaries.

        Starts with the left boundary if available, then the right boundary.
        """
        lane = self.map.segments[start]
        left_boundary: parser.LaneBoundary | None = lane.left_boundary
        if left_boundary:
            yield from self._traverse_boundary(
                left_boundary,
                interp_distance=interp_distance,
            )

        right_boundary: parser.LaneBoundary | None = lane.right_boundary
        if right_boundary:
            yield from self._traverse_boundary(
                right_boundary,
                interp_distance=interp_distance,
            )

    def _traverse_boundary(
        self,
        boundary: parser.LaneBoundary,
        *,
        interp_distance: float | None = None,
    ) -> Iterable[tuple[int, int, EdgeType]]:
        """Traverse a lane boundary and yield edges for its nodes.

        This method will add nodes to the graph and yield edges, but edges are
        not added to the adjacency list directly.
        """
        edge_type = boundary.get_edge_type()
        if edge_type == EdgeType.NONE:
            edge_type = self.assume_none_as

        for i in range(len(boundary.nodes) - 1):
            src = boundary.nodes[i]
            dst = boundary.nodes[i + 1]

            yield from self.interpolate_edge(
                src,
                dst,
                edge_type=edge_type,
                interp_distance=interp_distance,
            )
