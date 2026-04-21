"""Map-graph builder for the Lyft Level 5 dataset."""

from __future__ import annotations

from pathlib import Path

from typing_extensions import override

from dronalize.core.categories import EdgeType
from dronalize.datasets.lyft.maps import parser
from dronalize.processing.maps.builder import BaseMapBuilder


class LyftMapBuilder(BaseMapBuilder):
    """Builder for a map graph from a Lyft LVL5 map.

    Parameters
    ----------
    lyft_map : parser.LyftLVL5Map
        Parsed Lyft Level 5 map data used to build the graph.
    """

    def __init__(self, lyft_map: parser.LyftLVL5Map) -> None:
        super().__init__()
        self.map: parser.LyftLVL5Map = lyft_map
        self.edge_map: dict[EdgeType, EdgeType] = {EdgeType.NONE: EdgeType.VIRTUAL}

        self.added_lanes: set[str] = set()

    @classmethod
    def from_files(cls, map_path: Path | str, meta_json: Path | str) -> LyftMapBuilder:
        """Create a map builder from map and metadata files.

        The metafile needs the `world_to_ecef` transformation matrix

        Parameters
        ----------
        map_path : Path or str
            Path to the proto file containing the map data.
        meta_json : Path or str
            Path to the JSON file containing metadata.

        Returns
        -------
        LyftMapBuilder
            A `LyftMapBuilder` instance initialized with the map.

        """
        if isinstance(map_path, str):
            map_path = Path(map_path)
        if isinstance(meta_json, str):
            meta_json = Path(meta_json)
        lyft_map = parser.LyftLVL5Map(map_path, meta_json)
        return cls(lyft_map)

    @override
    def build_impl(
        self, min_distance: float | None = None, interp_distance: float | None = None
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
            points=boundary.points,
            edge_type=[boundary.get_edge_type_from_src(i) for i in range(len(boundary.points) - 1)],
        )

    def _traverse_lane(self, lane: parser.Lane) -> None:
        """Traverse a lane and yield edges for its boundaries."""
        for boundary in (lane.left_boundary, lane.right_boundary):
            self.add_path_lazy(
                points=boundary.points,
                edge_type=[
                    boundary.get_edge_type_from_src(i) for i in range(len(boundary.points) - 1)
                ],
            )
