"""Map-graph builder for the highD dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import EdgeType
from dronalize.processing.maps.builder import FeatureMapBuilder
from dronalize.processing.maps.features import PathFeature

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class HighDMapBuilder(FeatureMapBuilder):
    """Map builder for the HighD dataset."""

    def __init__(self, meta_file: Path, start_x: float, end_x: float) -> None:
        self._start_x: float = start_x
        self._end_x: float = end_x
        self._meta_file: Path = meta_file

    @override
    def iter_features(self) -> Iterable[PathFeature]:
        data = pl.read_csv(self._meta_file).select(
            pl.col("upperLaneMarkings").str.split(";").cast(pl.List(pl.Float64)),
            pl.col("lowerLaneMarkings").str.split(";").cast(pl.List(pl.Float64)),
        )

        n_lane_markings = len(data["upperLaneMarkings"][0])
        for i, y in enumerate(data["upperLaneMarkings"][0]):
            yield PathFeature(
                points=((self._start_x, y), (self._end_x, y)),
                edge_types=(
                    EdgeType.ROAD_BORDER
                    if i == 0 or i == n_lane_markings - 1
                    else EdgeType.LINE_THIN_DASHED
                ),
            )

        n_lane_markings = len(data["lowerLaneMarkings"][0])
        for i, y in enumerate(data["lowerLaneMarkings"][0]):
            yield PathFeature(
                points=((self._start_x, y), (self._end_x, y)),
                edge_types=(
                    EdgeType.ROAD_BORDER
                    if i == 0 or i == n_lane_markings - 1
                    else EdgeType.LINE_THIN_DASHED
                ),
            )
