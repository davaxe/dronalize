"""Loader implementation for the highD dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory
from dronalize.core.scene import POSITIONS_VELOCITY_ACCELERATION
from dronalize.datasets.highd.maps.builder import HighDMapBuilder
from dronalize.datasets.shared import utils
from dronalize.datasets.shared.levelx_loader import LevelXDataLoader

if TYPE_CHECKING:
    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import Scene, TrajectorySchema
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.maps.resolver import MapResolver
    from dronalize.processing.models import LoaderRequest


class HighDLoader(LevelXDataLoader):
    """Loader for the highD dataset."""

    def __init__(
        self,
        *,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        """Initialize the highD loader."""
        super().__init__(data_root=Path(data_root) / "data", request=request, resources=resources)

    @staticmethod
    @override
    def meta_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the metadata CSV."""
        return [
            pl.col("id"),
            pl.col("numLaneChanges").alias("lane_changes"),
            pl
            .col("class")
            .replace_strict({"Car": AgentCategory.CAR.value, "Truck": AgentCategory.TRUCK.value})
            .cast(pl.Int32())
            .alias("agent_category"),
        ]

    @staticmethod
    @override
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
            pl.col("laneId").alias("lane_id"),
        ]

    @staticmethod
    @override
    def meta_schema() -> pl.Schema:
        """Define the schema for the metadata CSV."""
        return _META_SCHEMA

    @staticmethod
    @override
    def track_schema() -> pl.Schema:
        """Define the schema for the track CSV."""
        return _TRACK_SCHEMA

    @staticmethod
    @override
    def location_id_select(meta_df: pl.DataFrame, path: Path) -> str:
        _ = meta_df
        return str(path)

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_ACCELERATION

    @override
    def map_resolver(self) -> MapResolver:
        def _resolver(scene: Scene) -> MapGraph | None:
            if scene.map_key is None or self.map_config is None:
                return None

            min_x = scene.frame.select(pl.col("x")).min().item()
            max_x = scene.frame.select(pl.col("x")).max().item()
            dist = max_x - min_x
            builder = HighDMapBuilder(Path(scene.map_key), min_x - dist * 0.1, max_x + dist * 0.1)
            map_graph = builder.build(self.map_config.min_distance, self.map_config.interp_distance)
            return utils.extract_based_on_scene(map_graph, scene, self.map_config.extraction)

        return _resolver


_META_SCHEMA: pl.Schema = pl.Schema({"id": pl.Int32, "numLaneChanges": pl.Int8})

_TRACK_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int32,
    "id": pl.Int32,
    "width": pl.Float64,
    "height": pl.Float64,
    "x": pl.Float64,
    "y": pl.Float64,
    "xVelocity": pl.Float64,
    "yVelocity": pl.Float64,
    "xAcceleration": pl.Float64,
    "yAcceleration": pl.Float64,
})
