"""Loader implementation for the SinD dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory
from dronalize.core.polars_ops import yaw_from_pos_expr
from dronalize.core.scene import CANONICAL
from dronalize.processing.loading.base import BaseSceneLoader
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.processing.maps.resolver import MapResolver, no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.models import LoaderRequest


class SindLoader(BaseSceneLoader):
    """Loader for the SinD dataset."""

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        """Initialize the SinD loader."""
        super().__init__(data_root=data_root, request=request, resources=resources)

    @classmethod
    @override
    def unified_factory(
        cls,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> SindLoader:
        return cls(data_root, request, resources)

    @override
    def discover_sources(self) -> Iterable[Source[Path]]:
        for region in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for data_dir in sorted(p for p in region.iterdir() if p.is_dir()):
                yield Source(identifier=data_dir.name, data=data_dir, map_key=str(region.name))

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        subdir = source.data
        pedestrian_data_path = subdir / "Ped_smoothed_tracks.csv"
        vehicle_data_path = subdir / "Veh_smoothed_tracks.csv"
        vehicle_df = pl.scan_csv(vehicle_data_path, schema_overrides=_VEHICLE_SCHEMA).select(
            pl.col("track_id").alias("id"),
            pl.col("frame_id").alias("frame"),
            pl
            .col("agent_type")
            .replace_strict({
                "motorcycle": AgentCategory.MOTORCYCLE.value,
                "car": AgentCategory.CAR.value,
                "truck": AgentCategory.TRUCK.value,
                "tricycle": AgentCategory.TRICYCLE.value,
                "bus": AgentCategory.BUS.value,
                "bicycle": AgentCategory.BICYCLE.value,
            })
            .alias("agent_category"),
            pl.col("heading_rad").alias("yaw"),
            *("x", "y", "vx", "vy", "ax", "ay"),
        )

        pedestrian_df = pl.scan_csv(
            pedestrian_data_path, schema_overrides=_PEDESTRIAN_SCHEMA
        ).select(
            pl
            .col("track_id")
            .str.slice(1)
            .cast(pl.Int32)
            .add(vehicle_df.select(pl.col("id").max()).collect().item() + 1)
            .alias("id"),
            pl.col("frame_id").alias("frame"),
            pl
            .col("agent_type")
            .replace_strict({
                "pedestrian": AgentCategory.PEDESTRIAN.value,
                "animal": AgentCategory.ANIMAL.value,
            })
            .alias("agent_category"),
            yaw_from_pos_expr().alias("yaw"),
            *("x", "y", "vx", "vy", "ax", "ay"),
        )

        yield LoadedSourceData(pl.concat([vehicle_df, pedestrian_df]))

    @override
    def num_sources(self) -> int | None:
        if not self.root.is_dir():
            return 0
        return sum(1 for path in self.root.iterdir() if path.is_dir())

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return CANONICAL

    @override
    def map_resolver(self) -> MapResolver:
        shared_maps = self.resources.shared_maps
        if not shared_maps or self.map_config is None:
            return no_map()
        return shared_map(shared_maps)


_VEHICLE_SCHEMA = pl.Schema({
    "track_id": pl.Int32,
    "frame_id": pl.Int32,
    "agent_type": pl.Utf8,
    "heading_rad": pl.Float64,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "ax": pl.Float64,
    "ay": pl.Float64,
})

_PEDESTRIAN_SCHEMA = pl.Schema({
    "track_id": pl.Utf8,
    "frame_id": pl.Int32,
    "agent_type": pl.Utf8,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "ax": pl.Float64,
    "ay": pl.Float64,
})
