"""Loader implementation for the A43 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory
from dronalize.core.scene import POSITIONS_VELOCITY_ACCELERATION, Scene
from dronalize.datasets.a43.maps.builder import A43MapBuilder
from dronalize.datasets.shared import utils
from dronalize.processing.loading.base import BaseSceneLoader
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.processing.loading.options import DatasetOptionsModel
from dronalize.processing.maps.resolver import MapResolver

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.maps.resolver import MapResolver


class A43LoaderOptions(DatasetOptionsModel):
    rows_per_source: int = 40_000


class A43Loader(BaseSceneLoader[tuple[Path, int], A43LoaderOptions]):
    """Scene loader for the A43 dataset."""

    _dt: ClassVar[float] = 0.1
    _eps: ClassVar[float] = 1e-9

    @override
    def iter_sources(self) -> Iterable[Source[tuple[Path, int]]]:
        for i, csv_file in enumerate(self.root.glob("*.csv")):
            rows: int = pl.scan_csv(csv_file, infer_schema=False).select(pl.len()).collect().item()
            for j in range(0, rows, self.loader_options.rows_per_source):
                yield Source(identifier=i, data=(csv_file, j), map_key=csv_file.stem)

    @override
    def load_source(self, source: Source[tuple[Path, int]]) -> Iterable[LoadedSourceData]:
        path, i = source.data
        yield LoadedSourceData(
            pl
            .scan_csv(path, skip_rows=i, n_rows=self.loader_options.rows_per_source, schema=_SCHEMA)
            .with_columns(t0=pl.col("tseconds").min())
            .with_columns(
                frame=(
                    (((pl.col("tseconds") - pl.col("t0")) / self._dt) + 0.5 + self._eps)
                    .floor()
                    .cast(pl.Int64)
                )
            )
            .select(
                pl.col("ID").alias("id"),
                pl.col("frame"),
                *("x", "y", "vy", "vx", "ax", "ay"),
                pl
                .col("VehicleCategory")
                .replace_strict({
                    "Motorcycle": AgentCategory.MOTORCYCLE,
                    "Passenger Car": AgentCategory.CAR,
                    "Semi-trailer truck": AgentCategory.TRUCK,
                    "Truck": AgentCategory.TRUCK,
                    "Van": AgentCategory.VAN,
                    "Bus": AgentCategory.BUS,
                })
                .alias("agent_category"),
            )
        )

    @override
    def count_sources(self) -> int | None:
        total, rows_per_source = 0, self.loader_options.rows_per_source
        for csv_file in self.root.glob("*.csv"):
            rows: int = pl.scan_csv(csv_file, infer_schema=False).select(pl.len()).collect().item()
            total += (rows + rows_per_source - 1) // rows_per_source
        return total

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
            builder = A43MapBuilder(scene.map_key, min_x, max_x)
            map_graph = builder.build(self.map_config.min_distance, self.map_config.interp_distance)
            return utils.extract_based_on_scene(map_graph, scene, self.map_config.extraction)

        return _resolver


_SCHEMA: pl.Schema = pl.Schema({
    "tseconds": pl.Float64,
    "ttimestamp": pl.Utf8,
    "ID": pl.Int64,
    "VehicleCategory": pl.Utf8,
    "Length": pl.Float64,
    "Width": pl.Float64,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "ax": pl.Float64,
    "ay": pl.Float64,
})
