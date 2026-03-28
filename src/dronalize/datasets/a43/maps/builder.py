from __future__ import annotations

from typing_extensions import override

from dronalize.core.maps.edge_types import EdgeType
from dronalize.processing.maps.builder import BaseMapBuilder


class A43MapBuilder(BaseMapBuilder):
    """Graph builder for the A43 dataset.

    This dataset does not have actual map data, but the map can be reconstructed
    (inferred/estimated) from the trajectories of the vehicles. The graph
    builder uses a mathematical heuristic to infer the lane structure based on
    lane width and the number of lanes.

    """

    def __init__(self, data_file_name: str, min_x: float, max_x: float) -> None:
        super().__init__()
        self._markings: dict[str, list[float]] = {
            "DroneDataEastToWestCSV_220725": [-3.75, 0, 3.75, 7.5],
            "DroneDataWestToEastCSV_220725": [-7.5, -3.75, 0, 3.75, 7.5],
        }
        if data_file_name not in self._markings:
            msg = f"Unknown data file name: {data_file_name}"
            raise ValueError(msg)

        self._name: str = data_file_name
        self._min_x: float = min_x
        self._max_x: float = max_x

    @override
    def build_impl(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        markings = self._markings[self._name]
        for i, y in enumerate(markings):
            if i in {0, len(markings) - 1}:
                edge_type = EdgeType.ROAD_BORDER
            else:
                edge_type = EdgeType.LINE_THIN_DASHED

            self.add_path_lazy(
                points=[(self._min_x, y), (self._max_x, y)],
                edge_type=edge_type,
            )
