from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config.map import MapConfig
from dronalize.datasets.common import utils
from dronalize.datasets.common.levelx_loader import LevelXDataLoader
from dronalize.datasets.highd.map.builder import HighDMapBuilder
from dronalize.scene import POSITIONS_VELOCITY_ACCELERATION_V1

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.config import LoaderConfig
    from dronalize.maps.graph import MapGraph
    from dronalize.maps.resolver import MapResolver
    from dronalize.scene import Scene, SceneSchema


class HighDLoader(LevelXDataLoader):
    """Loader for the highD dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        """Initialize the trajectory data loader for the highD dataset.

        Parameters
        ----------
        data_root : Path
            Path to the root directory of the highD dataset, which should contain a "data"
        loader_config : , optional
            Loader configuration. If None, the default configuration is used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            Path(data_root) / "data",
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
        )

    @staticmethod
    @override
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
        return str(path)

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_VELOCITY_ACCELERATION_V1

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.no_extraction(interp_distance=10)

    @override
    def map_resolver(self) -> MapResolver:
        def _resolver(scene: Scene) -> MapGraph | None:
            if scene.map_key is None:
                return None

            min_x = scene.inner.select(pl.col("x")).min().item()
            max_x = scene.inner.select(pl.col("x")).max().item()
            dist = max_x - min_x
            builder = HighDMapBuilder(Path(scene.map_key), min_x - dist * 0.1, max_x + dist * 0.1)
            map_graph = builder.build(self.map_config.min_distance, self.map_config.interp_distance)
            return utils.extract_based_on_scene(map_graph, scene, self.map_config.extraction)

        return _resolver


_META_SCHEMA: pl.Schema = pl.Schema({
    "id": pl.Int32,
    "numLaneChanges": pl.Int8,
})

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
