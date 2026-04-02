"""Loader implementation for the ExiD dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.datasets.shared.levelx_loader import LevelXDataLoader

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.categories import DatasetSplit
    from dronalize.processing.ingest.config import LoaderConfig
    from dronalize.processing.ingest.splits import SplitConfig
    from dronalize.processing.maps.config import MapConfig


class ExiDLoader(LevelXDataLoader):
    """Trajectory data loader for the exiD dataset.

    The exiD (extracted from Drone) dataset was recorded at highway exits and
    entries in Germany. It contains naturalistic vehicle trajectories extracted
    from drone footage, covering a variety of traffic participants including
    cars, trucks, buses, and motorcycles interacting at on-ramps and off-ramps.

    This loader inherits all trajectory processing logic from
    `XLevelDataLoader`, as the exiD dataset follows the same CSV format used
    across the X-level dataset family.

    """

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
    ) -> None:
        """Initialize the exiD loader."""
        data_root = Path(data_root)
        super().__init__(
            data_root / "data",
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )

    @staticmethod
    @override
    def track_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the track CSV."""
        select = LevelXDataLoader.track_data_select()
        select.append(pl.col("laneletId").alias("lane_id"))
        return select

    @staticmethod
    @override
    def track_schema() -> pl.Schema:
        """Define the schema for the track CSV."""
        return _TRACK_SCHEMA

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return super().default_config().with_highway(required_lane_changes=3, negative_keep_every=3)


_TRACK_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int32,
    "trackId": pl.Int32,
    "xCenter": pl.Float64,
    "yCenter": pl.Float64,
    "xVelocity": pl.Float64,
    "yVelocity": pl.Float64,
    "xAcceleration": pl.Float64,
    "yAcceleration": pl.Float64,
    "laneletId": pl.Int32,
})


if __name__ == "__main__":
    from dronalize.datasets.exid import DESCRIPTOR
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env

    root = resolve_dataset_root_from_env("exid")
    _ = debug_descriptor(DESCRIPTOR, root)
