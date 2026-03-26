from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.maps.builder import BaseMapBuilder
from dronalize.maps.edge_type import EdgeType

if TYPE_CHECKING:
    from pathlib import Path


class HighDMapBuilder(BaseMapBuilder):
    """Map builder for the HighD dataset.

    The data only contains the y coordinates of the lane markings, so we create
    the nodes at specified start and end x coordinates. The lane markings are
    represented as edges between the nodes. The outermost lanes are classified as
    road borders.

    """

    def __init__(self, meta_file: Path, start_x: float, end_x: float) -> None:
        """Initialize the map builder.

        Parameters
        ----------
        meta_file : Path
            Path to the meta file containing the lane markings.
        start_x : float
            The x coordinate of the start of the road section.
        end_x : float
            The x coordinate of the end of the road section.

        """
        self._start_x: float = start_x
        self._end_x: float = end_x
        self._meta_file: Path = meta_file
        super().__init__()

    @override
    def build_impl(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        # These are used implicitly if `MapBuilder.build` is called.
        _min_distance, _interp_distance = min_distance, interp_distance
        data = pl.read_csv(self._meta_file).select(
            pl.col("upperLaneMarkings").str.split(";").cast(pl.List(pl.Float64)),
            pl.col("lowerLaneMarkings").str.split(";").cast(pl.List(pl.Float64)),
        )

        n_lane_markings = len(data["upperLaneMarkings"][0])
        for i, y in enumerate(data["upperLaneMarkings"][0]):
            self.add_path_lazy(
                [(self._start_x, y), (self._end_x, y)],
                EdgeType.ROAD_BORDER
                if i == 0 or i == n_lane_markings - 1
                else EdgeType.LINE_THIN_DASHED,
            )

        n_lane_markings = len(data["lowerLaneMarkings"][0])
        for i, y in enumerate(data["lowerLaneMarkings"][0]):
            self.add_path_lazy(
                [(self._start_x, y), (self._end_x, y)],
                EdgeType.ROAD_BORDER
                if i == 0 or i == n_lane_markings - 1
                else EdgeType.LINE_THIN_DASHED,
            )
