"""Map-graph builder for the US-101 dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from dronalize.datasets.shared.highway_builder import HighwayLaneMapBuilder, LaneDescription

if TYPE_CHECKING:
    from pathlib import Path


class US101MapBuilder(HighwayLaneMapBuilder):
    """Map builder for the US101 dataset."""

    def __init__(self, data_dir: Path) -> None:
        files = list(data_dir.rglob("trajectories*.csv"))
        data = pl.scan_csv(files).select(
            pl.col("Vehicle_ID").alias("id"),
            pl.col("Local_X").alias("x").mul(0.3048),
            pl.col("Local_Y").alias("y").mul(0.3048),
            pl.col("Lane_ID").alias("lane_id"),
        )
        super().__init__(
            data,
            bin_size=10.0,
            include_outer_borders=True,
            smoothing=3.0,
            lane_description=LaneDescription(ids=list(range(1, 9)), direction=[True] * 8),
        )
