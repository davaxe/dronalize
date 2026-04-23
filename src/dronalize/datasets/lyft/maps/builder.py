"""Map-graph builder for the Lyft Level 5 dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.datasets.lyft.maps import parser
from dronalize.processing.maps.builder import FeatureMapBuilder
from dronalize.processing.maps.features import PathFeature

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class LyftMapBuilder(FeatureMapBuilder):
    """Builder for a map graph from a Lyft LVL5 map."""

    def __init__(self, lyft_map: parser.LyftLVL5Map) -> None:
        self.map: parser.LyftLVL5Map = lyft_map

    @classmethod
    def from_files(cls, map_path: Path, meta_json: Path) -> LyftMapBuilder:
        """Create a map builder from map and metadata files."""
        return cls(parser.LyftLVL5Map(map_path, meta_json))

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
    def _lane_features(lane: parser.Lane, *, junction: bool) -> Iterable[PathFeature]:
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
