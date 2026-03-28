from __future__ import annotations

from itertools import chain, repeat
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from dronalize.core.maps.edge_types import EdgeType
from dronalize.datasets.argoverse1.maps import parser, utils
from dronalize.processing.maps.builder import BaseMapBuilder, Point

if TYPE_CHECKING:
    from pathlib import Path


class Argoverse1MapBuilder(BaseMapBuilder):
    """A builder for creating a graph representation of an Argoverse1 map."""

    class SegmentEndpoint(TypedDict):
        """Endpoints of a lane segment used for graph building."""

        start_id: int
        end_id: int

    class SegmentEndpoints(TypedDict):
        """Endpoints of a lane segment with a line string."""

        right: Argoverse1MapBuilder.SegmentEndpoint
        left: Argoverse1MapBuilder.SegmentEndpoint

    def __init__(self, argoverse_map: parser.Argoverse1Map) -> None:
        """Initialize the map builder with an `Argoverse1Map`."""
        if not argoverse_map.is_parsed():
            argoverse_map.parse()

        super().__init__()
        self.lane_segments: dict[int, parser.LaneSegment] = argoverse_map.lane_segments
        self.map_nodes: dict[int, Point] = argoverse_map.nodes

        self.max_distance_between_connections: float = 1.0

    @classmethod
    def from_xml_file(cls, path: Path) -> Argoverse1MapBuilder:
        """Create a map builder from an Argoverse 1 XML file."""
        argoverse_map = parser.Argoverse1Map.from_xml_file(path)
        return cls(argoverse_map)

    @override
    def build_impl(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        # Dict to store endpoints (start and end) of each lane segment to enable
        # efficient connection of segments after all segments are added
        self._lane_endpoints: dict[
            int,
            Argoverse1MapBuilder.SegmentEndpoints,
        ] = {}
        self._build_segment_edges(
            interp_distance=interp_distance,
            min_distance=min_distance,
        )
        self._connect_lane_segments()

    # --- Private methods ---

    def _connect_lane_segments(self) -> None:
        """Connect endpoints of all connected lane segments in the graph.

        The connections are based on the predecessors and successors of each
        lane segment.
        """
        already_connected: dict[int, set[int]] = {}
        for segment in self.lane_segments.values():
            _ = already_connected.setdefault(segment.id, set())
            for pre_id in segment.predecessors:
                if pre_id in already_connected[segment.id]:
                    continue

                self._connect_endpoints(pre_id, segment.id)
                already_connected[segment.id].add(pre_id)
                already_connected.setdefault(pre_id, set()).add(segment.id)

            for suc_id in segment.successors:
                if suc_id in already_connected[segment.id]:
                    continue

                self._connect_endpoints(segment.id, suc_id)
                already_connected[segment.id].add(suc_id)
                already_connected.setdefault(suc_id, set()).add(segment.id)

    def _connect_endpoints(
        self,
        endpoint_from: int,
        endpoint_to: int,
        left_edge_type: EdgeType = EdgeType.VIRTUAL,
        right_edge_type: EdgeType = EdgeType.VIRTUAL,
    ) -> None:
        """Connect two lane segment endpoints in the graph.

        Parameters
        ----------
        endpoint_from : int
            Lane segment ID from which to connect.
        endpoint_to : int
            Lane segment ID to which to connect.
        left_edge_type : EdgeType, optional
            Left border edge type.
        right_edge_type : EdgeType, optional
            Right border edge type.

        """
        if endpoint_from not in self._lane_endpoints or endpoint_to not in self._lane_endpoints:
            return

        from_endpoints = self._lane_endpoints[endpoint_from]
        to_endpoints = self._lane_endpoints[endpoint_to]

        right_from = from_endpoints["right"]["end_id"]
        left_from = from_endpoints["left"]["end_id"]
        right_to = to_endpoints["right"]["start_id"]
        left_to = to_endpoints["left"]["start_id"]

        if (
            not self.edge_exists(right_from, right_to, right_edge_type)
            and self.node_distance(right_from, right_to) < self.max_distance_between_connections
        ):
            self.add_edge(right_from, right_to, right_edge_type)

        if (
            not self.edge_exists(left_from, left_to, left_edge_type)
            and self.node_distance(left_from, left_to) < self.max_distance_between_connections
        ):
            self.add_edge(left_from, left_to, left_edge_type)

    def _build_segment_edges(
        self,
        *,
        interp_distance: float | None = None,
        min_distance: float | None = None,
    ) -> None:
        """Build edges through a lane segment."""
        for segment in self.lane_segments.values():
            self._add_boundary_edges(
                segment,
                interp_distance=interp_distance,
                min_distance=min_distance,
            )

    def _add_boundary_edges(
        self,
        segment: parser.LaneSegment,
        *,
        interp_distance: float | None = None,
        min_distance: float | None = None,
    ) -> None:
        """Add edges for the left and right boundaries of a lane segment."""
        node_ids = list(segment.node_ids)
        centerline = [self.map_nodes[i] for i in node_ids]
        l_type, r_type = EdgeType.VIRTUAL, EdgeType.VIRTUAL
        add_as_polygon = (
            utils.lane_segment_is_regulatory(segment)
            and segment.turn_direction == parser.TurnType.NONE
        )
        add_as_polygon &= not utils.any_lane_segment_is_regulatory(
            utils.lane_segment_successors(segment, self.lane_segments),
        )

        right, left = utils.edge_borders_from_centerline(np.array(centerline))
        if add_as_polygon:
            polygon = np.vstack([left, right[::-1], left[0]])
            edge_types = chain(
                repeat(l_type, len(right) - 1),  # The left border types
                [EdgeType.REGULATORY],  # Type between left and right border
                repeat(r_type, len(left) - 1),  # The right border types
                [EdgeType.REGULATORY],  # Type between right and left border
            )
            added_nodes = self._numpy_add_edge_loop(
                polygon,
                edge_type=list(edge_types),
                interp_distance=interp_distance,
                min_distance=min_distance,
            )
            l_nodes = added_nodes[: len(left) - 1]
            r_nodes = added_nodes[len(left) : -1]
        else:
            # Add left border edges
            l_nodes = self._numpy_add_edge_loop(
                left,
                edge_type=l_type,
                interp_distance=interp_distance,
                min_distance=min_distance,
            )
            # Add right border edges
            r_nodes = self._numpy_add_edge_loop(
                right,
                edge_type=r_type,
                interp_distance=interp_distance,
                min_distance=min_distance,
            )

        # Store endpoints for the lane segment for later use to connect lanes
        if len(l_nodes) > 1 and len(r_nodes) > 1:
            self._lane_endpoints[segment.id] = {
                "right": {
                    "start_id": r_nodes[0],
                    "end_id": r_nodes[-1],
                },
                "left": {
                    "start_id": l_nodes[0],
                    "end_id": l_nodes[-1],
                },
            }

    def _numpy_add_edge_loop(
        self,
        numpy_nodes: npt.NDArray[np.float64],
        edge_type: EdgeType | list[EdgeType] = EdgeType.VIRTUAL,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> list[int]:
        if len(numpy_nodes) == 0:
            return []

        # Handle edge types
        if isinstance(edge_type, EdgeType):
            edge_type_list = list(repeat(edge_type, times=len(numpy_nodes) - 1))
        else:
            edge_type_list = edge_type

        # 1. Create the starting node
        prev_pt: Point = (float(numpy_nodes[0][0]), float(numpy_nodes[0][1]))
        prev_id = self.add_node(*prev_pt)

        added_ids = [prev_id]
        min_distance = min_distance if min_distance is not None else 0.0

        i, j = 0, 1
        edge_idx = 0
        while i < len(numpy_nodes) - 1:
            src_vec, dst_vec = numpy_nodes[i], numpy_nodes[j]
            distance = np.linalg.norm(dst_vec - src_vec)
            if distance < min_distance and j < len(numpy_nodes) - 1:
                j += 1
                continue

            dst_pt: Point = (float(dst_vec[0]), float(dst_vec[1]))
            dst_id = self.add_node(*dst_pt)
            current_edge_type = edge_type_list[min(edge_idx, len(edge_type_list) - 1)]

            last_id = prev_id
            for s_id, d_id, e_type in self.interpolate_edge(
                prev_id,
                prev_pt,
                dst_id,
                dst_pt,
                interp_distance,
                current_edge_type,
            ):
                self.add_edge(s_id, d_id, e_type)
                added_ids.append(d_id)
                last_id = d_id

            prev_pt = dst_pt
            prev_id = last_id

            # Advance indices
            i = j
            j = i + 1
            edge_idx += 1

        return added_ids
