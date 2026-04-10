"""Loader implementation for the ExiD dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.datasets.shared.levelx_loader import LevelXDataLoader

if TYPE_CHECKING:
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.models import LoaderRequest


class ExiDLoader(LevelXDataLoader):
    """Loader for the ExiD dataset."""

    def __init__(
        self,
        *,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        """Initialize the ExiD loader."""
        super().__init__(data_root=Path(data_root) / "data", request=request, resources=resources)

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
