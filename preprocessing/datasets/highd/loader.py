from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from preprocessing.common.loaders.xlevel import XLevelDataLoader
from preprocessing.core import AgentCategory

if TYPE_CHECKING:
    from preprocessing.core import LoaderConfig


class HighDLoader(XLevelDataLoader):
    """Processor for the highD dataset."""

    def __init__(
        self,
        data_dir: Path,
        config: LoaderConfig | None = None,
        lane_change_ratio: float | None = 1.0,
    ) -> None:
        """Initialize the trajectory data loader for the highD dataset.

        It is possible to rebalance the dataset by adjusting the number of lane
        changing agents compared to non-lane changing agents. This can be done
        by setting the `lane_change_ratio` parameter. For example, a ratio of
        0.5 would result in half as many lane changing agents as non-lane
        changing agents. Typically highway datasets are heavily imbalanced
        towards non-lane changing agents, which means that a high ratio con
        result in way less total data.

        Parameters
        ----------
        data_dir : Path
            Path to the directory containing the .csv data files.
        config : LoaderConfig, optional
            Processor configuration. If None, default configuration will be used.
        lane_change_ratio : float, optional
            Ratio to rebalance lane changing vs non-lane changing agents.

        """
        super().__init__(data_dir, config)
        # Update internal state to enable rebalancing of lane changing vs non-lane changing agents
        self._rebalance_ratio = lane_change_ratio

    @staticmethod
    def meta_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the metadata CSV."""
        return [
            pl.col("id"),
            pl.col("numLaneChanges").alias("lane_changes"),
            pl
            .col("class")
            .replace_strict({
                "Car": AgentCategory.CAR.value,
                "Truck": AgentCategory.TRUCK.value,
            })
            .alias("agent_category"),
        ]

    @staticmethod
    def track_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the track CSV."""
        return [
            pl.col("frame"),
            pl.col("id"),
            pl.col("x").add(pl.col("width") / 2),
            pl.col("y").add(pl.col("height") / 2),
            pl.col("xVelocity").alias("vx"),
            pl.col("yVelocity").alias("vy"),
            pl.col("xAcceleration").alias("ax"),
            pl.col("yAcceleration").alias("ay"),
        ]

    @staticmethod
    def meta_schema() -> pl.Schema:
        """Define the schema for the metadata CSV."""
        return _META_SCHEMA

    @staticmethod
    def track_schema() -> pl.Schema:
        """Define the schema for the track CSV."""
        return _TRACK_SCHEMA


_META_SCHEMA: pl.Schema = pl.Schema({
    "id": pl.Int32,
    "numLaneChanges": pl.Int8,
})

_TRACK_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int32,
    "id": pl.Int32,
    "width": pl.Float32,
    "height": pl.Float32,
    "x": pl.Float32,
    "y": pl.Float32,
    "xVelocity": pl.Float32,
    "yVelocity": pl.Float32,
    "xAcceleration": pl.Float32,
    "yAcceleration": pl.Float32,
})

if __name__ == "__main__":
    data_dir = Path("/home/west/Developer/behavior-prediction/datasets/highD/data")

    processor = HighDLoader(data_dir=data_dir)
    count = 0
    for scene in processor.scenes():
        if count % 200 == 0:
            print(scene.map_context)
            print(f"Processed {count} scenes")
        count += 1
    print(f"Total scenes processed: {count}")
