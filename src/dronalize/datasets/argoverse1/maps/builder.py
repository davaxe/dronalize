"""Map-graph builder for the Argoverse 1 dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from dronalize.core.categories import EdgeType
from dronalize.datasets.argoverse1.maps import parser
from dronalize.processing.maps.builder import FeatureMapBuilder, Point
from dronalize.processing.maps.features import PathFeature

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
    def iter_features(self) -> Iterable[PathFeature]:
        for segment in self._lane_segments.values():
            yield from self._segment_features(segment)
        yield from self._connection_features()

    def _segment_features(self, segment: parser.LaneSegment) -> Iterable[PathFeature]:
        node_ids = list(segment.node_ids)
        centerline = [self._map_nodes[i] for i in node_ids]
        left_key = f"lane:{segment.id}:left"
        right_key = f"lane:{segment.id}:right"

        add_as_polygon = (
            lane_segment_is_regulatory(segment) and segment.turn_direction == parser.TurnType.NONE
        )
        add_as_polygon &= not any_lane_segment_is_regulatory(
            lane_segment_successors(segment, self._lane_segments)
        )

        right, left = edge_borders_from_centerline(np.array(centerline))
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

    def _get_segment_borders(self, segment_id: int) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
        """Get the left and right border points for a segment."""
        segment = self._lane_segments[segment_id]
        node_ids = list(segment.node_ids)
        centerline = [self._map_nodes[i] for i in node_ids]

        right, left = edge_borders_from_centerline(np.array(centerline))
        left_points = tuple((float(x), float(y)) for x, y in left)
        right_points = tuple((float(x), float(y)) for x, y in right)
        return (left_points, right_points)

    def _connection_features(self) -> Iterable[PathFeature]:
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
    ) -> Iterable[PathFeature]:
        from_left, from_right = self._get_segment_borders(endpoint_from)
        to_left, to_right = self._get_segment_borders(endpoint_to)

        # Check distance constraints before yielding
        max_dist = self._max_distance_between_connections
        right_dist_sq = (from_right[-1][0] - to_right[0][0]) ** 2 + (
            from_right[-1][1] - to_right[0][1]
        ) ** 2
        left_dist_sq = (from_left[-1][0] - to_left[0][0]) ** 2 + (
            from_left[-1][1] - to_left[0][1]
        ) ** 2

        if right_dist_sq <= max_dist**2:
            yield PathFeature(
                points=(from_right[-1], to_right[0]), edge_types=right_edge_type, min_distance=0.0
            )

        if left_dist_sq <= max_dist**2:
            yield PathFeature(
                points=(from_left[-1], to_left[0]), edge_types=left_edge_type, min_distance=0.0
            )


# `swap_left_and_right` and `edge_borders_from_centerline` are utility functions
# that are taken (with some modifications) from the original Argoverse 1
# codebase. Available at: https://github.com/argoverse/argoverse-api
# Note: `edge_borders_from_centerline` was originally named `centerline_to_polygon`.


def swap_left_and_right(
    condition: npt.NDArray[np.bool_],
    left_centerline: npt.NDArray[np.float64],
    right_centerline: npt.NDArray[np.float64],
) -> Iterable[npt.NDArray[np.float64]]:
    """Swap points in left and right centerline according to condition.

    Parameters
    ----------
    condition : np.ndarray
        Boolean array of shape (N,). Where True, swap the values in the left
        and right centerlines.
    left_centerline : np.ndarray
        The left centerline, whose points should be swapped with the right
        centerline.
    right_centerline : np.ndarray
        The right centerline.

    Returns
    -------
    left_centerline : np.ndarray
        The (possibly swapped) left centerline.
    right_centerline : np.ndarray
        The (possibly swapped) right centerline.

    """
    right_swap_indices = right_centerline[condition]
    left_swap_indices = left_centerline[condition]

    left_centerline[condition] = right_swap_indices
    right_centerline[condition] = left_swap_indices
    return left_centerline, right_centerline


def edge_borders_from_centerline(
    centerline: npt.NDArray[np.float64], width_scaling_factor: float = 1.0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert a lane centerline polyline into a rough polygon of the lane's area.

    On average, a lane is 3.8 meters in width. Thus, we allow 1.9 m on each
    side. We use this as the length of the hypotenuse of a right triangle, and
    compute the other two legs to find the scaled x and y displacement.

    Parameters
    ----------
    centerline : np.ndarray, shape (N, 2)
        The lane centerline polyline.
    width_scaling_factor : float, optional
        Multiplier that scales 3.8 meters to get the lane width.

    Returns
    -------
    right_centerline : np.ndarray
        Right border of the lane.
    left_centerline : np.ndarray
        Left border of the lane.

    """
    # eliminate duplicates
    _, inds = np.unique(centerline, axis=0, return_index=True)
    # does not return indices in sorted order
    inds = np.sort(inds)
    centerline = centerline[inds]

    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])

    # compute the normal at each point
    slopes = dy / dx
    inv_slopes = -1.0 / slopes

    thetas = np.arctan(inv_slopes)
    x_disp = 3.8 * width_scaling_factor / 2.0 * np.cos(thetas)
    y_disp = 3.8 * width_scaling_factor / 2.0 * np.sin(thetas)

    displacement = np.hstack([x_disp[:, np.newaxis], y_disp[:, np.newaxis]])
    right_centerline = centerline + displacement
    left_centerline = centerline - displacement

    # right centerline position depends on sign of dx and dy
    subtract_cond1 = np.logical_and(dx > 0, dy < 0)
    subtract_cond2 = np.logical_and(dx > 0, dy > 0)
    subtract_cond = np.logical_or(subtract_cond1, subtract_cond2)
    left_centerline, right_centerline = swap_left_and_right(
        subtract_cond, left_centerline, right_centerline
    )

    # right centerline also depended on if we added or subtracted y
    neg_disp_cond = displacement[:, 1] > 0
    left_centerline, right_centerline = swap_left_and_right(
        neg_disp_cond, left_centerline, right_centerline
    )

    left_centerline, right_centerline = right_centerline, left_centerline

    # return the polygon
    return right_centerline, left_centerline


def lane_segment_successors(
    lane_segment: parser.LaneSegment, lane_segments: dict[int, parser.LaneSegment]
) -> Iterable[parser.LaneSegment]:
    """Get successors of a lane segment."""
    return (lane_segments[lane_segment_id] for lane_segment_id in lane_segment.successors)


def lane_segment_predecessors(
    lane_segment: parser.LaneSegment, lane_segments: dict[int, parser.LaneSegment]
) -> Iterable[parser.LaneSegment]:
    """Get predecessors of a lane segment."""
    return (lane_segments[lane_segment_id] for lane_segment_id in lane_segment.predecessors)


def lane_segment_is_regulatory(lane_segment: parser.LaneSegment) -> bool:
    """Check if a lane segment is regulatory.

    Parameters
    ----------
    lane_segment : LaneSegment
        The lane segment to check.

    Returns
    -------
    bool
        True if the lane segment is regulatory, False otherwise.

    """
    return lane_segment.is_intersection and lane_segment.has_traffic_control


def any_lane_segment_is_regulatory(lane_segments: Iterable[parser.LaneSegment]) -> bool:
    """Check if any lane segment in the iterable is regulatory."""
    return any(lane_segment_is_regulatory(lane_segment) for lane_segment in lane_segments)
