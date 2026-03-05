from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from dronalize.common.map.lane import HighWayLaneGraphBuilder, LaneDescription

if TYPE_CHECKING:
    from pathlib import Path


class I80GraphBuilder(HighWayLaneGraphBuilder):
    """Graph builder for the I-80 dataset.

    This dataset do not have acutal map data, but the map can be reconstructed
    (infered/estimated) from the trajectories of the vehicles. The graph builder
    uses a simple heuristic to infer the lane structure, see
    `HighWayLaneGraphBuilder` for details.

    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize the graph builder.

        Parameters
        ----------
        data_dir : Path
            The root directory of the I-80 dataset.

        """
        files = list(data_dir.rglob("trajectories*.csv"))
        data = pl.scan_csv(files).select(
            pl.col("Vehicle_ID").alias("id"),
            pl.col("Local_X").alias("x").mul(0.3048),  # Convert feet to meters
            pl.col("Local_Y").alias("y").mul(0.3048),
            pl.col("Lane_ID").alias("lane_id"),
        )
        super().__init__(
            data,
            bin_size=10.0,
            include_outer_borders=True,
            smoothing=3.0,
        )
        self._lane_description = LaneDescription(ids=list(range(1, 7)), direction=[True] * 6)
