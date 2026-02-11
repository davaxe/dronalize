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

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterable

    from preprocessing.datasets.argoverse1.map.parser import LaneSegment

# `swap_left_and_right` and `edge_borders_from_centerline` are utility functions
# that are taken (with some modifications) from the original Argoverse 1
# codebase. Available at: https://github.com/argoverse/argoverse-api
# Note: `edge_borders_from_centerline` was originally named `centerline_to_polygon`.


def swap_left_and_right(
    condition: np.ndarray,
    left_centerline: np.ndarray,
    right_centerline: np.ndarray,
) -> Iterable[np.ndarray]:
    """Swap points in left and right centerline according to condition.

    Args:
       condition: Numpy array of shape (N,) of type boolean. Where true, swap
            the values in the left and right centerlines.
       left_centerline: The left centerline, whose points should be swapped with
            the right centerline.
       right_centerline: The right centerline.

    Returns:
       left_centerline
       right_centerline

    """
    right_swap_indices = right_centerline[condition]
    left_swap_indices = left_centerline[condition]

    left_centerline[condition] = right_swap_indices
    right_centerline[condition] = left_swap_indices
    return left_centerline, right_centerline


def edge_borders_from_centerline(
    centerline: np.ndarray,
    width_scaling_factor: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert a lane centerline polyline into a rough polygon of the lane's area.

    On average, a lane is 3.8 meters in width. Thus, we allow 1.9 m on each
    side. We use this as the length of the hypotenuse of a right triangle, and
    compute the other two legs to find the scaled x and y displacement.

    Args:
       centerline: Numpy array of shape (N,2).
       width_scaling_factor: Multiplier that scales 3.8 meters to get the lane width.

    Returns:
       polygon: Numpy array of shape (2N+1,2), with duplicate first and last
        vertices.

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

    Args:
        lane_segment: The lane segment to check.

    Returns:
        True if the lane segment is regulatory, False otherwise.

    """
    return lane_segment.is_intersection and lane_segment.has_traffic_control


def any_lane_segment_is_regulatory(
    lane_segments: Iterable[LaneSegment],
) -> bool:
    """Check if any lane segment in the iterable is regulatory."""
    return any(lane_segment_is_regulatory(lane_segment) for lane_segment in lane_segments)
