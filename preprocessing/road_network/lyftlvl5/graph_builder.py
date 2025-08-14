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

from pathlib import Path
from typing import TYPE_CHECKING

from preprocessing.road_network.common import (
    GraphBuilder,
    IntIDNode,
    MapGraph,
    check_interpolation_params,
)
from preprocessing.road_network.edge_type import EdgeType
from preprocessing.road_network.lyftlvl5 import parser

if TYPE_CHECKING:
    from collections.abc import Iterable


class LyftLVL5MapGraphBuilder(GraphBuilder[int, IntIDNode]):
    """Builder for a map graph from a Lyft LVL5 map."""

    def __init__(self, lyft_map: parser.LyftLVL5Map) -> None:
        """Initialize the graph builder with a Lyft LVL5 map."""
        super().__init__()
        self.map = lyft_map

        self.added_lanes: set[str] = set()

    @classmethod
    def from_files(
        cls,
        map_path: Path | str,
        meta_json: Path | str,
    ) -> LyftLVL5MapGraphBuilder:
        """Create a graph builder from map and metadata files.

        The metafile needs the `world_to_ecef` transformation matrix

        Args:
            map_path: path to the proto file containing the map data.
            meta_json: path to the JSON file containing metadata.

        Returns:
            A `LyftLVL5MapGraphBuilder` instance initialized with the map.

        """
        if isinstance(map_path, str):
            map_path = Path(map_path)
        if isinstance(meta_json, str):
            meta_json = Path(meta_json)
        lyft_map = parser.LyftLVL5Map(map_path, meta_json)
        return cls(lyft_map)

    def new_node(self, x: float, y: float, z: float = 0) -> IntIDNode:
        """Create a new node with the given coordinates."""
        return IntIDNode(x, y, z)

    def build(
        self,
        *,
        interpolate: bool = False,
        interp_distance: float | None = None,
    ) -> MapGraph:
        """Build the map graph from the Lyft LVL5 map.

        Args:
            interpolate: true if edges should be interpolated, false otherwise.
            interp_distance: the approximate distance between interpolated
                nodes. If None, no interpolation is performed.

        Returns:
            A `MapGraph` object representing the map graph.

        """
        check_interpolation_params(
            interpolate=interpolate,
            interp_distance=interp_distance,
        )

        self._add_road_segment_edges(interp_distance=interp_distance)
        self._add_junction_edges(interp_distance=interp_distance)

        return self.build_graph()

    # --- Private methods ---

    def _add_junction_edges(
        self,
        *,
        interp_distance: float | None = None,
    ) -> None:
        """Add edges for junctions in the map."""
        for junction in self.map.junctions.values():
            for lane_id in junction.extra_lanes:
                if lane_id in self.added_lanes:
                    continue

                self.add_edges_from_iterable(
                    self._traverse_junction_lane(
                        self.map.lanes[lane_id],
                        interp_distance=interp_distance,
                    ),
                )
                self.added_lanes.add(lane_id)

    def _add_road_segment_edges(
        self,
        *,
        interp_distance: float | None = None,
    ) -> None:
        """Add edges for road segments in the map."""
        for road_segment in self.map.road_network_segments.values():
            for lane_id in road_segment.lanes_iter():
                if lane_id in self.added_lanes:
                    continue

                self.add_edges_from_iterable(
                    self._traverse_lane(
                        self.map.lanes[lane_id],
                        interp_distance=interp_distance,
                    ),
                )
                self.added_lanes.add(lane_id)

    def _traverse_junction_lane(
        self,
        lane: parser.Lane,
        *,
        interp_distance: float | None = None,
        edge_type: EdgeType = EdgeType.VIRTUAL,
    ) -> Iterable[tuple[int, int, EdgeType]]:
        """Traverse a junction lane and yield edges for its boundaries."""
        yield from self._traverse_boundary(
            boundary=lane.left_boundary,
            interp_distance=interp_distance,
            alt_edge_type=edge_type,
        )

    def _traverse_lane(
        self,
        lane: parser.Lane,
        *,
        interp_distance: float | None = None,
    ) -> Iterable[tuple[int, int, EdgeType]]:
        """Traverse a lane and yield edges for its boundaries."""
        yield from self._traverse_boundary(
            boundary=lane.left_boundary,
            interp_distance=interp_distance,
            alt_edge_type=EdgeType.VIRTUAL,
        )

        yield from self._traverse_boundary(
            boundary=lane.right_boundary,
            interp_distance=interp_distance,
            alt_edge_type=EdgeType.VIRTUAL,
        )

    def _traverse_boundary(
        self,
        boundary: parser.LaneBoundary,
        *,
        interp_distance: float | None = None,
        alt_edge_type: EdgeType,
    ) -> Iterable[tuple[int, int, EdgeType]]:
        """Traverse a lane boundary and yield edges for its nodes."""
        for i in range(len(boundary.nodes) - 1):
            src_node = boundary.nodes[i]
            dst_node = boundary.nodes[i + 1]

            edge_type = boundary.get_edge_type_from_src(i)
            if edge_type == EdgeType.NONE:
                edge_type = alt_edge_type

            yield from self.interpolate_edge(
                src_node,
                dst_node,
                interp_distance=interp_distance,
                edge_type=edge_type,
            )
