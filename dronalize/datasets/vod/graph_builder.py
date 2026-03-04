from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from scipy.interpolate import BSpline
from typing_extensions import override

from dronalize.core import EdgeType
from dronalize.datasets.nuscenes.map.graph_builder import NuScenesMapGraphBuilder
from dronalize.datasets.nuscenes.map.parser import NuScenesMap

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.core.protocols.graph_builder import Point


class VODMapGraphBuilder(NuScenesMapGraphBuilder):
    """A builder for creating a MapGraph from a VOD map."""

    def __init__(
        self,
        path: Path,
        *,
        ignore_edge_types: set[str] | None = None,
    ) -> None:
        """Initialize the VODMapGraphBuilder.

        Parameters
        ----------
        path : Path
            Path to the VOD map JSON file.
        ignore_edge_types : set[str], optional
            A set of edge type names to ignore during graph construction.

        """
        nuscenes_map = NuScenesMap(path)
        self.ignore_edge_types = ignore_edge_types if ignore_edge_types is not None else set()

        super().__init__(nuscenes_map)
        self.lane_polygon_edge: EdgeType | None = EdgeType.LINE_THIN
        self._edge_type_methods = {
            "road_divider": self._add_road_divider_edges,
            "walkway": self._add_walkway_edges,
            "pedestrian_crossing": self._add_pedestrian_crossing_edges,
            "traffic_light": self._add_traffic_light_edges,
            "stop_line": self._add_stop_line_edges,
            "lane": self._add_lane_edges,
            "carpark": self._add_carpark_edges,
        }

    @override
    @classmethod
    def from_json_file(
        cls,
        path: Path,
        *,
        ignore_edge_types: set[str] | None = None,
    ) -> Self:
        """Create a VODMapGraphBuilder from a file path."""
        return cls(path, ignore_edge_types=ignore_edge_types)

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
            points: list[Point] = list(zip(x.tolist(), y.tolist(), strict=True))
            self.add_node_edges_loop_min_dist(
                points,
                min_distance=self.min_distance,
                interp_distance=None,
                edge_type=EdgeType.STOP,
                is_polygon=False,
            )
