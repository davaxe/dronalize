from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.common.loaders.xlevel import XLevelDataLoader
from dronalize.core import AgentCategory, LoaderConfig
from dronalize.core.datatypes import map_context as mc
from dronalize.core.protocols.loader import Source

if TYPE_CHECKING:
    from collections.abc import Iterable


class AD4CHELoader(XLevelDataLoader):
    """Processor for the AD4CHE dataset."""

    def __init__(
        self,
        data_dir: Path,
        config: LoaderConfig | None = None,
        lane_change_ratio: float | None = 1.0,
    ) -> None:
        """Initialize the trajectory data loader for the AD4CHE dataset.

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

    @override
    def sources(self) -> Iterable[Source[int, pl.LazyFrame]]:
        for i, subdir in enumerate(self._data_dir.iterdir(), start=1):
            recording_meta = subdir / f"{i:0>2}_recordingMeta.csv"
            meta = subdir / f"{i:0>2}_tracksMeta.csv"
            tracks = subdir / f"{i:0>2}_tracks.csv"

            recording_meta_data = pl.read_csv(recording_meta)
            location_id = recording_meta_data.select(pl.col("locationId")).item()
            meta_df = pl.scan_csv(meta, schema_overrides=self.meta_schema()).select(
                *self.meta_data_select()
            )

            tracks_df = pl.scan_csv(tracks, schema_overrides=self.track_schema()).select(
                *self.track_data_select()
            )
            combined = tracks_df.join(meta_df, left_on="id", right_on="id")
            yield Source(
                identifier=i,
                inner=combined,
                map_context=mc.Explicit(id=str(location_id)),
            )

    @override
    def num_sources(self) -> int | None:
        num_files: int = sum(1 for p in self._data_dir.iterdir())
        return num_files // 4 - 1

    @staticmethod
    def meta_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the metadata CSV."""
        return [
            pl.col("id"),
            pl.col("numLaneChanges").alias("lane_changes"),
            pl
            .col("class")
            .replace_strict({
                "car": AgentCategory.CAR.value,
                "truck": AgentCategory.TRUCK.value,
                "bus": AgentCategory.BUS.value,
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

    @override
    def default_config(self) -> LoaderConfig:
        return (
            LoaderConfig(60, 150, 1 / 30)
            .resampling_parameters(1, 3)
            .scene_filtering_parameters()
            .window_parameters(45)
        )


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
    from dronalize.common.plotting import plot_trajectories

    data_dir = Path("C:/Users/axdai/dev/python/dronalize/data/ad4che")

    processor = AD4CHELoader(data_dir=data_dir)
    count = 0
    for scene in processor.scenes():
        if count % 50 == 0:
            plt = plot_trajectories(scene.inner, group_by="id", width=1200, height=800).show()
            print(scene.map_context)
            print(f"Processed {count} scenes")
        count += 1
    print(f"Total scenes processed: {count}")
