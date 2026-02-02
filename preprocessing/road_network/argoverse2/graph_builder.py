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

from typing import TYPE_CHECKING, Literal

from matplotlib import pyplot as plt

from preprocessing.road_network.argoverse2 import parser
from preprocessing.road_network.common import (
    GraphBuilder,
    IntIDNode,
    MapGraph,
)
from preprocessing.road_network.edge_type import EdgeType

if TYPE_CHECKING:
    from pathlib import Path


class Argoverse2GraphBuilder(GraphBuilder[int, IntIDNode]):
    """A builder for creating a graph representation of an Argoverse2 map."""

    def __init__(self, map_data: parser.Argoverse2Map) -> None:
        """Initialize the graph builder with an Argoverse2Map."""
        super().__init__()
        self.map = map_data

        self.edge_map: dict[EdgeType, EdgeType] = {EdgeType.NONE: EdgeType.VIRTUAL}

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
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> MapGraph:
        """Build a `MapGraph` from the `Argoverse2Map`.

        To perform interpolation, set `interpolate` to True and provide a value
        for `interp_distance`.

        Args:
            min_distance: the minimum distance between nodes when adding edges.
                If None, no minimum distance is enforced.
            interp_distance: the target distance for interpolation. If None,
                no interpolation is performed.

        Returns:
            A `MapGraph` object containing the node positions, edge indices, and
            edge types.

        """
        self._add_lane_boundary_edges()
        self._add_pedestrian_crossing_edges()

        return self.build_graph(
            min_dist=min_distance,
            interp_distance=interp_distance,
        )

    # --- Private methods ----

    def _add_pedestrian_crossing_edges(self) -> None:
        """Add edges for pedestrian crossings in the map."""
        for crossing in self.map.pedestrian_crossings.values():
            self.add_path_lazy(
                nodes=crossing.first_edge,
                edge_type=EdgeType.PEDESTRIAN_MARKING,
            )
            self.add_path_lazy(
                nodes=crossing.second_edge,
                edge_type=EdgeType.PEDESTRIAN_MARKING,
            )

    def _add_lane_boundary_edges(self) -> None:
        """Add edges for lane boundaries in the map."""
        lane_segments = self.map.segments.copy()

        while lane_segments:
            _, segment = lane_segments.popitem()
            left = self._lane_segment_nodes(segment, side="left")
            if left is not None:
                nodes, edge = left
                self.add_path_lazy(
                    nodes=nodes,
                    edge_type=edge,
                )

            right_nodes = self._lane_segment_nodes(segment, side="right")
            if right_nodes is not None:
                nodes, edge = right_nodes
                self.add_path_lazy(
                    nodes=nodes,
                    edge_type=edge,
                )

    def _lane_segment_nodes(
        self,
        segment: parser.LaneSegment,
        side: Literal["left", "right"],
    ) -> tuple[list[IntIDNode], EdgeType] | None:
        """Get all nodes for a lane segment's boundaries."""
        boundary = (
            segment.left_boundary if side == "left" else segment.right_boundary
        )
        if boundary is None:
            return None

        nodes = boundary.nodes
        edge_type = boundary.get_edge_type()
        return nodes, edge_type


if __name__ == "__main__":
    from pathlib import Path

    # Example usage
    json_path = Path(
        "../datasets/av2/test/407afee5-aa3d-43cc-a258-1ffa52ee735f/log_map_archive_407afee5-aa3d-43cc-a258-1ffa52ee735f.json"
    )
    builder = Argoverse2GraphBuilder.from_json_file(json_path)
    graph = builder.build(
        min_distance=5.0,
        interp_distance=None,
    )

    print(f"Number of nodes: {len(graph.node_types)}")

    graph.plot_graph()
    plt.savefig("test_argoverse2_graph.pdf")
