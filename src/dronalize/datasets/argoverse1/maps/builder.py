"""Map-graph builder for the Argoverse 1 dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from dronalize.core.categories import EdgeType
from dronalize.datasets.argoverse1.maps import parser, utils
from dronalize.processing.maps.builder import FeatureMapBuilder, Point
from dronalize.processing.maps.features import EndpointLinkFeature, PathFeature

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class Argoverse1MapBuilder(FeatureMapBuilder):
    """A builder for creating a graph representation of an Argoverse1 map."""

    def __init__(self, argoverse_map: parser.Argoverse1Map) -> None:
        if not argoverse_map.is_parsed():
            argoverse_map.parse()

        self._lane_segments: dict[int, parser.LaneSegment] = argoverse_map.lane_segments
        self._map_nodes: dict[int, Point] = argoverse_map.nodes
        self._max_distance_between_connections: float = 1.0

    @classmethod
    def from_xml_file(cls, path: Path) -> Argoverse1MapBuilder:
        """Create a map builder from an Argoverse 1 XML file."""
        return cls(parser.Argoverse1Map.from_xml_file(path))

    @override
    def iter_features(self) -> Iterable[PathFeature | EndpointLinkFeature]:
        for segment in self._lane_segments.values():
            yield from self._segment_features(segment)
        yield from self._connection_features()

    def _segment_features(self, segment: parser.LaneSegment) -> Iterable[PathFeature]:
        node_ids = list(segment.node_ids)
        centerline = [self._map_nodes[i] for i in node_ids]
        left_key = f"lane:{segment.id}:left"
        right_key = f"lane:{segment.id}:right"

        add_as_polygon = (
            utils.lane_segment_is_regulatory(segment)
            and segment.turn_direction == parser.TurnType.NONE
        )
        add_as_polygon &= not utils.any_lane_segment_is_regulatory(
            utils.lane_segment_successors(segment, self._lane_segments)
        )

        right, left = utils.edge_borders_from_centerline(np.array(centerline))
        left_points = tuple((float(x), float(y)) for x, y in left)
        right_points = tuple((float(x), float(y)) for x, y in right)

        yield PathFeature(points=left_points, edge_types=EdgeType.VIRTUAL, key=left_key)
        yield PathFeature(points=right_points, edge_types=EdgeType.VIRTUAL, key=right_key)

        if add_as_polygon:
            yield PathFeature(
                points=(left_points[-1], right_points[-1]),
                edge_types=EdgeType.REGULATORY,
                min_distance=0.0,
            )
            yield PathFeature(
                points=(right_points[0], left_points[0]),
                edge_types=EdgeType.REGULATORY,
                min_distance=0.0,
            )

    def _connection_features(self) -> Iterable[EndpointLinkFeature]:
        already_connected: dict[int, set[int]] = {}
        for segment in self._lane_segments.values():
            _ = already_connected.setdefault(segment.id, set())
            for predecessor_id in segment.predecessors:
                if predecessor_id in already_connected[segment.id]:
                    continue
                yield from self._connect_endpoints(predecessor_id, segment.id)
                already_connected[segment.id].add(predecessor_id)
                already_connected.setdefault(predecessor_id, set()).add(segment.id)

            for successor_id in segment.successors:
                if successor_id in already_connected[segment.id]:
                    continue
                yield from self._connect_endpoints(segment.id, successor_id)
                already_connected[segment.id].add(successor_id)
                already_connected.setdefault(successor_id, set()).add(segment.id)

    def _connect_endpoints(
        self,
        endpoint_from: int,
        endpoint_to: int,
        left_edge_type: EdgeType = EdgeType.VIRTUAL,
        right_edge_type: EdgeType = EdgeType.VIRTUAL,
    ) -> Iterable[EndpointLinkFeature]:
        yield EndpointLinkFeature(
            src_key=f"lane:{endpoint_from}:right",
            dst_key=f"lane:{endpoint_to}:right",
            edge_type=right_edge_type,
            max_distance=self._max_distance_between_connections,
        )
        yield EndpointLinkFeature(
            src_key=f"lane:{endpoint_from}:left",
            dst_key=f"lane:{endpoint_to}:left",
            edge_type=left_edge_type,
            max_distance=self._max_distance_between_connections,
        )
