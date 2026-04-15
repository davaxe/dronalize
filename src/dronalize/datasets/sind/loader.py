"""Loader implementation for the SinD dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory
from dronalize.core.scene import CANONICAL
from dronalize.processing.loading.base import BaseSceneLoader, LoaderSplitCapabilities
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

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_source_split=True
    )

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
        data_root: Path = self.root / "data"
        if not data_root.is_dir():
            return
        for subdir in sorted(p for p in data_root.iterdir() if p.is_dir()):
            yield Source(
                identifier=subdir.name,
                data=subdir,
                map_key=str(self._resolve_map_key(subdir.parent.name)),
            )

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        subdir = source.data
        print(f"Loading source from {subdir}...")
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
            pl.lit(None).alias("yaw"),
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
        if not isinstance(shared_maps, dict):
            return no_map()
        return shared_map(shared_maps)

    @staticmethod
    def _resolve_map_key(path_name: str) -> str:
        if path_name.startswith("changchun"):
            return "changchun"
        if path_name.startswith("xian"):
            return "xian"
        if "NR" in path_name:
            return "nr_ll2"
        return "map_relink_law_save"


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
