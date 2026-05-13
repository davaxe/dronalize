"""Map-graph builder for the A43 dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.core.categories import EdgeType
from dronalize.processing.maps import FeatureMapBuilder, PathFeature

if TYPE_CHECKING:
    from collections.abc import Iterable


class A43MapBuilder(FeatureMapBuilder):
    """Graph builder for the A43 dataset."""

    def __init__(self, data_file_name: str, min_x: float, max_x: float) -> None:
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
    def iter_features(self) -> Iterable[PathFeature]:
        markings = self._markings[self._name]
        for i, y in enumerate(markings):
            edge_type = (
                EdgeType.ROAD_BORDER if i in {0, len(markings) - 1} else EdgeType.LINE_THIN_DASHED
            )
            yield PathFeature(points=((self._min_x, y), (self._max_x, y)), edge_types=edge_type)
