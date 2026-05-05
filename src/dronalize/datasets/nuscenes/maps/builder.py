"""Map builder abstraction for `NuScenesMap`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Self, override

from dronalize.core.categories import EdgeType
from dronalize.datasets.nuscenes.maps import parser
from dronalize.processing.maps.builder import FeatureMapBuilder
from dronalize.processing.maps.features import PathFeature, Point

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path


class NuScenesMapBuilder(FeatureMapBuilder):
    """A builder for creating a MapGraph from a NuscenesMap."""

    def __init__(
        self,
        nuscenes_map: parser.NuScenesMap,
        ignore_edge_types: set[str] | None = None,
        *,
        lane_polygon_edge: EdgeType | None = None,
    ) -> None:
        self.map: parser.NuScenesMap = nuscenes_map
        self.map_nodes: dict[str, parser.Node] = self.map.nodes
        self.lane_polygon_edge: EdgeType | None = lane_polygon_edge
        self.ignore_edge_types: set[str] = set() if ignore_edge_types is None else ignore_edge_types
        self._edge_type_methods: dict[str, Callable[[], Iterable[PathFeature]]] = {
            "road_divider": self._road_divider_features,
            "lane_divider": self._lane_divider_features,
            "walkway": self._walkway_features,
            "pedestrian_crossing": self._pedestrian_crossing_features,
            "traffic_light": self._traffic_light_features,
            "stop_line": self._stop_line_features,
            "lane": self._lane_features,
            "carpark": self._carpark_features,
        }

    @classmethod
    def from_json_file(cls, path: Path, *, ignore_edge_types: set[str] | None = None) -> Self:
        """Create a map builder from a file path."""
        return cls(parser.NuScenesMap(path), ignore_edge_types=ignore_edge_types)

    @override
    def iter_features(self) -> Iterable[PathFeature]:
        for edge_type, method in self._edge_type_methods.items():
            if edge_type in self.ignore_edge_types:
                continue
            yield from method()

    def _node_points(self, node_ids: list[str]) -> list[Point]:
        return [self.map_nodes[i].as_point() for i in node_ids]

    def _road_divider_features(self) -> Iterable[PathFeature]:
        for road_divider in self.map.road_dividers.values():
            line: parser.Line = self.map.lines[road_divider.line]
            yield PathFeature(
                points=tuple(self._node_points(line.nodes)), edge_types=EdgeType.LINE_THICK
            )

    def _lane_divider_features(self) -> Iterable[PathFeature]:
        for lane_divider in self.map.lane_dividers.values():
            points, edge_types = self._extract_edges(lane_divider.segment_types)
            yield PathFeature(points=tuple(points), edge_types=tuple(edge_types))

    def _lane_features(self) -> Iterable[PathFeature]:
        for lane in self.map.lanes.values():
            if lane.left_lane_divider_segments:
                points, edge_types = self._extract_edges(lane.left_lane_divider_segments)
                yield PathFeature(points=tuple(points), edge_types=tuple(edge_types))
            if lane.right_lane_divider_segments:
                points, edge_types = self._extract_edges(lane.right_lane_divider_segments)
                yield PathFeature(points=tuple(points), edge_types=tuple(edge_types))
            if self.lane_polygon_edge is not None:
                lane_polygon: parser.Polygon = self.map.polygons[lane.polygon]
                yield PathFeature(
                    points=tuple(self._node_points(lane_polygon.exterior_nodes)),
                    edge_types=self.lane_polygon_edge,
                    closed=True,
                )

    def _walkway_features(self) -> Iterable[PathFeature]:
        for walkway in self.map.walkways.values():
            polygon: parser.Polygon = self.map.polygons[walkway.polygon]
            yield PathFeature(
                points=tuple(self._node_points(polygon.exterior_nodes)),
                edge_types=EdgeType.CURB,
                closed=True,
                min_distance=0.0,
            )

    def _pedestrian_crossing_features(self) -> Iterable[PathFeature]:
        for crossing in self.map.pedestrian_crossings.values():
            polygon: parser.Polygon = self.map.polygons[crossing.polygon]
            yield PathFeature(
                points=tuple(self._node_points(polygon.exterior_nodes)),
                edge_types=EdgeType.PEDESTRIAN_MARKING,
                closed=True,
                min_distance=0.0,
            )

    def _traffic_light_features(self) -> Iterable[PathFeature]:
        for traffic_light in self.map.traffic_lights.values():
            line: parser.Line = self.map.lines[traffic_light.line]
            yield PathFeature(
                points=tuple(self._node_points(line.nodes)),
                edge_types=EdgeType.REGULATORY,
                min_distance=0.0,
            )

    def _stop_line_features(self) -> Iterable[PathFeature]:
        for stop_line in self.map.stop_lines.values():
            if not stop_line.is_valid(allow_pedestrian_crossings=False):
                continue
            polygon: parser.Polygon = self.map.polygons[stop_line.polygon]
            yield PathFeature(
                points=tuple(self._node_points(polygon.exterior_nodes)),
                edge_types=EdgeType.STOP,
                closed=True,
                min_distance=0.0,
            )

    def _carpark_features(self) -> Iterable[PathFeature]:
        for carpark in self.map.carpark_areas.values():
            polygon: parser.Polygon = self.map.polygons[carpark.polygon]
            yield PathFeature(
                points=tuple(self._node_points(polygon.exterior_nodes)),
                edge_types=EdgeType.VIRTUAL,
                closed=True,
                min_distance=0.0,
            )

    def _extract_edges(
        self, segments: list[tuple[str, parser.SegmentDividerType]]
    ) -> tuple[list[Point], list[EdgeType]]:
        return (
            [self.map_nodes[n_id].as_point() for n_id, _ in segments],
            [parser.SegmentDividerType.to_edge_type(s_type) for _, s_type in segments[:-1]],
        )
