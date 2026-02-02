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

from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
from scipy.interpolate import BSpline

from preprocessing.road_network import nuscenes
from preprocessing.road_network.edge_type import EdgeType

if TYPE_CHECKING:
    from pathlib import Path

# TODO:
# - Add centerlines correctly by using `arcline_path_3` available in the VOD map


class VODMapGraphBuilder(nuscenes.NuScenesMapGraphBuilder):
    """A builder for creating a MapGraph from a VOD map."""

    def __init__(self, path: Path, *, debug_parsing: bool = False) -> None:
        """Initialize the VODMapGraphBuilder.

        Args:
            path: Path to the VOD map JSON file.
            debug_parsing: If True, enables debug prints for parsing issues.

        """
        nuscenes_map = nuscenes.NuScenesMap(path, enable_debug_prints=debug_parsing)

        super().__init__(nuscenes_map)
        self.lane_polygon_edge: None | EdgeType = EdgeType.LINE_THIN
        self.edge_type_methods = {
            "road_divider": self._add_road_divider_edges,
            "walkway": self._add_walkway_edges,
            "pedestrian_crossing": self._add_pedestrian_crossing_edges,
            "traffic_light": self._add_traffic_light_edges,
            "stop_line": self._add_stop_line_edges,
            "lane": self._add_lane_edges,
            "carpark": self._add_carpark_edges,
            # "centerline": self._add_centerline_edges,
        }

    @classmethod
    def from_json_file(
        cls,
        path: Path,
        *,
        debug_parsing: bool = False,
    ) -> Self:
        """Create a NuscenesMapGraphBuilder from a file path."""
        return cls(path, debug_parsing=debug_parsing)

    def _add_centerline_edges(self, _interp_distance: float | None) -> None:
        """Add edges for arcline paths."""
        for arcline in self.map.arcline_path_3.values():
            ctrl = np.asarray(arcline.ctrl, dtype=float)
            spline = BSpline(
                arcline.knots,
                ctrl,
                3,
                extrapolate=False,
            )

            u = np.linspace(0, 1, 100)
            xy = spline(u).astype(np.float64)
            x, y = xy[:, 0], xy[:, 1]
            nodes = [self.new_node(xi, yi) for xi, yi in zip(x, y, strict=True)]
            self.add_node_edges_loop_min_dist(
                nodes,
                min_distance=self.min_distance,
                interp_distance=None,
                edge_type=EdgeType.STOP,
                is_polygon=False,
            )
