from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.datasets.argoverse1.maps.parser import LaneSegment

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
    centerline: npt.NDArray[np.float64],
    width_scaling_factor: float = 1.0,
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
        subtract_cond,
        left_centerline,
        right_centerline,
    )

    # right centerline also depended on if we added or subtracted y
    neg_disp_cond = displacement[:, 1] > 0
    left_centerline, right_centerline = swap_left_and_right(
        neg_disp_cond,
        left_centerline,
        right_centerline,
    )

    left_centerline, right_centerline = right_centerline, left_centerline

    # return the polygon
    return right_centerline, left_centerline


def lane_segment_successors(
    lane_segment: LaneSegment,
    lane_segments: dict[int, LaneSegment],
) -> Iterable[LaneSegment]:
    """Get successors of a lane segment."""
    return (lane_segments[lane_segment_id] for lane_segment_id in lane_segment.successors)


def lane_segment_predecessors(
    lane_segment: LaneSegment,
    lane_segments: dict[int, LaneSegment],
) -> Iterable[LaneSegment]:
    """Get predecessors of a lane segment."""
    return (lane_segments[lane_segment_id] for lane_segment_id in lane_segment.predecessors)


def lane_segment_is_regulatory(
    lane_segment: LaneSegment,
) -> bool:
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


def any_lane_segment_is_regulatory(
    lane_segments: Iterable[LaneSegment],
) -> bool:
    """Check if any lane segment in the iterable is regulatory."""
    return any(lane_segment_is_regulatory(lane_segment) for lane_segment in lane_segments)
