from __future__ import annotations

from pathlib import Path

from typing_extensions import override

from dronalize.core.datatypes.categories import EdgeType
from dronalize.core.graph.builder import GraphBuilder
from dronalize.core.graph.nodes import IntIDNode
from dronalize.datasets.lyft.map import parser


class LyftMapGraphBuilder(GraphBuilder[int, IntIDNode]):
    """Builder for a map graph from a Lyft LVL5 map."""

    def __init__(self, lyft_map: parser.LyftLVL5Map) -> None:
        """Initialize the graph builder with a Lyft LVL5 map."""
        super().__init__()
        self.map = lyft_map
        self.edge_map: dict[EdgeType, EdgeType] = {EdgeType.NONE: EdgeType.VIRTUAL}

        self.added_lanes: set[str] = set()

    @classmethod
    def from_files(
        cls,
        map_path: Path | str,
        meta_json: Path | str,
    ) -> LyftMapGraphBuilder:
        """Create a graph builder from map and metadata files.

        The metafile needs the `world_to_ecef` transformation matrix

        Parameters
        ----------
        map_path : Path or str
            Path to the proto file containing the map data.
        meta_json : Path or str
            Path to the JSON file containing metadata.

        Returns
        -------
        LyftMapGraphBuilder
            A `LyftMapGraphBuilder` instance initialized with the map.

        """
        if isinstance(map_path, str):
            map_path = Path(map_path)
        if isinstance(meta_json, str):
            meta_json = Path(meta_json)
        lyft_map = parser.LyftLVL5Map(map_path, meta_json)
        return cls(lyft_map)

    @override
    def new_node(self, x: float, y: float, z: float = 0) -> IntIDNode:
        return IntIDNode(self.next_node_id(), x, y, z)

    @override
    def build_impl(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        # Used implicitly when `self.add_path_lazy` is called
        _, _ = min_distance, interp_distance
        self._add_road_segment_edges()
        self._add_junction_edges()

    # --- Private methods ---

    def _add_junction_edges(self) -> None:
        """Add edges for junctions in the map."""
        for junction in self.map.junctions.values():
            for lane_id in junction.extra_lanes:
                if lane_id in self.added_lanes:
                    continue

                self._traverse_junction_lane(self.map.lanes[lane_id])
                self.added_lanes.add(lane_id)

    def _add_road_segment_edges(self) -> None:
        """Add edges for road segments in the map."""
        for road_segment in self.map.road_network_segments.values():
            for lane_id in road_segment.lanes_iter():
                if lane_id in self.added_lanes:
                    continue

                self._traverse_lane(self.map.lanes[lane_id])
                self.added_lanes.add(lane_id)

    def _traverse_junction_lane(self, lane: parser.Lane) -> None:
        """Traverse a junction lane and yield edges for its boundaries."""
        boundary = lane.left_boundary
        self.add_path_lazy(
            nodes=boundary.nodes,
            edge_type=[boundary.get_edge_type_from_src(i) for i in range(len(boundary.nodes) - 1)],
        )

    def _traverse_lane(self, lane: parser.Lane) -> None:
        """Traverse a lane and yield edges for its boundaries."""
        for boundary in (lane.left_boundary, lane.right_boundary):
            self.add_path_lazy(
                nodes=boundary.nodes,
                edge_type=[
                    boundary.get_edge_type_from_src(i) for i in range(len(boundary.nodes) - 1)
                ],
            )
