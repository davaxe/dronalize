from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core import AgentCategory, LoaderConfig
from dronalize.core.protocols.loader import Source
from dronalize.datasets.common.xlevel_loader import XLevelDataLoader

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class AD4CHELoader(XLevelDataLoader):
    """Processor for the AD4CHE dataset."""

    def __init__(
        self,
        data_root: Path,
        loader_config: LoaderConfig | None = None,
        *,
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
        data_root : Path
            Path to the directory containing the .csv data files.
        loader_config : LoaderConfig, optional
            Processor configuration. If None, default configuration will be used.
        lane_change_ratio : float, optional
            Ratio to rebalance lane changing vs non-lane changing agents.

        """
        super().__init__(data_root / "AD4CHE_Data_V1.0", loader_config)
        # Update internal state to enable rebalancing of lane changing vs non-lane changing agents
        self._rebalance_ratio = lane_change_ratio

    @override
    def all_sources(self) -> Iterable[Source[int, Path]]:
        for i, subdir in enumerate(sorted(self._data_dir.iterdir()), start=1):
            recording_meta = subdir / f"{i:0>2}_recordingMeta.csv"
            recording_meta_data = pl.read_csv(recording_meta)
            location_id = recording_meta_data.select(pl.col("locationId")).item()
            yield Source(
                identifier=i,
                inner=subdir,
                map_key=str(location_id),
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

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(60, 150, 1 / 30)
            .with_resampling(1, 3)
            .with_filtering(require_frames=[59])
            .with_window(45)
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
    import time
    from pathlib import Path

    loader = AD4CHELoader(Path("data/ad4che"))
    cout = 0
    start = time.perf_counter()
    for _scene in loader.scenes():
        cout += 1

    end = time.perf_counter()
    print(f"Processed {cout} scenes in {end - start:.2f} seconds")
