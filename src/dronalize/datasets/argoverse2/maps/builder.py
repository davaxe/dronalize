"""Map-graph builder for the Argoverse 2 dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from typing_extensions import override

from dronalize.core.categories import EdgeType
from dronalize.datasets.argoverse2.maps import parser
from dronalize.processing.maps.builder import FeatureMapBuilder, Point
from dronalize.processing.maps.features import PathFeature

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


class Argoverse2MapBuilder(FeatureMapBuilder):
    """A builder for creating a graph representation of an Argoverse2 map."""

    def __init__(self, map_data: parser.Argoverse2Map) -> None:
        self.map: parser.Argoverse2Map = map_data

    @classmethod
    def from_json_file(cls, json_file: Path) -> Argoverse2MapBuilder:
        """Create a map builder from an Argoverse 2 JSON file."""
        map_data = parser.Argoverse2Map(json_file)
        return cls(map_data)

    @override
    def edge_remap(self) -> Mapping[EdgeType, EdgeType]:
        return {EdgeType.NONE: EdgeType.VIRTUAL}

    @override
    def iter_features(self) -> Iterable[PathFeature]:
        for crossing in self.map.pedestrian_crossings.values():
            yield PathFeature(
                points=tuple(crossing.first_edge), edge_types=EdgeType.PEDESTRIAN_MARKING
            )
            yield PathFeature(
                points=tuple(crossing.second_edge), edge_types=EdgeType.PEDESTRIAN_MARKING
            )

        for segment in self.map.segments.values():
            left = self._lane_segment_points(segment, side="left")
            if left is not None:
                points, edge = left
                yield PathFeature(points=tuple(points), edge_types=edge)

            right = self._lane_segment_points(segment, side="right")
            if right is not None:
                points, edge = right
                yield PathFeature(points=tuple(points), edge_types=edge)

    @staticmethod
    def _lane_segment_points(
        segment: parser.LaneSegment, side: Literal["left", "right"]
    ) -> tuple[list[Point], EdgeType] | None:
        boundary = segment.left_boundary if side == "left" else segment.right_boundary
        if boundary is None:
            return None

        return boundary.points, boundary.get_edge_type()
