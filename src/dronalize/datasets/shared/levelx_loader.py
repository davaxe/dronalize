from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import Self, override

from dronalize.core.categories import AgentCategory
from dronalize.core.scene import CANONICAL
from dronalize.datasets.shared import utils
from dronalize.processing.loading.base import BaseSceneLoader
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.processing.maps.resolver import MapResolver, no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.models import LoaderRequest


@dataclass(slots=True, frozen=True)
class SourceData:
    path: Path
    x0: float | None = None
    y0: float | None = None


class LevelXDataLoader(BaseSceneLoader[SourceData]):
    """Common trajectory data loader for X-level datasets.

    Each discovered source corresponds to one full recording. The source payload
    stays lightweight: it is just the shared dataset root plus a stable
    recording identifier.

    With no changes this supports: rounD, inD, exiD, uniD, and sinD.

    Parameters
    ----------
    data_root : Path or str
        Root directory containing the extracted recording CSV files.
    """

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        super().__init__(data_root=data_root, request=request, resources=resources)

    @classmethod
    @override
    def unified_factory(
        cls,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> Self:
        return cls(data_root=data_root, request=request, resources=resources)

    @staticmethod
    def meta_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the metadata CSV."""
        return [
            pl.col("trackId").alias("id"),
            pl
            .col("class")
            .replace_strict(
                {
                    "car": AgentCategory.CAR.value,
                    "truck": AgentCategory.TRUCK.value,
                    "bus": AgentCategory.BUS.value,
                    "trailer": AgentCategory.TRAILER.value,
                    "motorcycle": AgentCategory.MOTORCYCLE.value,
                    "bicycle": AgentCategory.BICYCLE.value,
                    "pedestrian": AgentCategory.PEDESTRIAN.value,
                    "van": AgentCategory.VAN.value,
                    "truck_bus": AgentCategory.TRUCK.value,
                    "animal": AgentCategory.ANIMAL.value,
                },
                return_dtype=pl.Int32,
            )
            .alias("agent_category"),
        ]

    @staticmethod
    def track_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the track CSV."""
        return [
            pl.col("frame"),
            pl.col("trackId").alias("id"),
            pl.col("xCenter").alias("x"),
            pl.col("yCenter").alias("y"),
            pl.col("xVelocity").alias("vx"),
            pl.col("yVelocity").alias("vy"),
            pl.col("xAcceleration").alias("ax"),
            pl.col("yAcceleration").alias("ay"),
            pl.col("heading").alias("yaw"),
        ]

    @staticmethod
    def location_id_select(meta_df: pl.DataFrame, path: Path) -> str:
        """Select the relevant columns from the recording metadata CSV."""
        _ = path  # Added path since highD wants to use  the path as key
        return meta_df.select(pl.col("locationId")).item()

    @staticmethod
    def meta_schema() -> pl.Schema:
        """Define the schema for the metadata CSV."""
        return _META_SCHEMA

    @staticmethod
    def track_schema() -> pl.Schema:
        """Define the schema for the track CSV."""
        return _TRACK_SCHEMA

    @override
    def iter_sources(self) -> Iterable[Source[SourceData]]:
        for recording_id in self._recording_ids():
            recording_meta = self.root / f"{recording_id:0>2}_recordingMeta.csv"
            recording_meta_data = pl.read_csv(recording_meta)
            location_id = self.location_id_select(recording_meta_data, recording_meta)
            columns = recording_meta_data.columns

            utm_x0: float | None = None
            utm_y0: float | None = None
            if "xUtmOrigin" in columns and "yUtmOrigin" in columns:
                utm_x0 = recording_meta_data.select(pl.col("xUtmOrigin")).item()
                utm_y0 = recording_meta_data.select(pl.col("yUtmOrigin")).item()

            yield Source(
                identifier=recording_id,
                data=SourceData(self.root, x0=utm_x0, y0=utm_y0),
                map_key=str(location_id),
            )

    @override
    def load_source(self, source: Source[SourceData]) -> Iterable[LoadedSourceData]:
        tracks = source.data.path / f"{source.identifier:0>2}_tracks.csv"
        meta = source.data.path / f"{source.identifier:0>2}_tracksMeta.csv"
        meta_df = pl.scan_csv(meta, schema_overrides=self.meta_schema()).select(
            *self.meta_data_select()
        )
        tracks_df = pl.scan_csv(tracks, schema_overrides=self.track_schema()).select(
            *self.track_data_select()
        )
        combined = tracks_df.join(meta_df, left_on="id", right_on="id")
        combined = combined.with_columns(
            (pl.col("x") + (source.data.x0 or 0.0)).alias("x"),
            (pl.col("y") + (source.data.y0 or 0.0)).alias("y"),
        )
        yield LoadedSourceData(combined)

    @override
    def count_sources(self) -> int | None:
        return len(self._recording_ids())

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return CANONICAL

    @override
    def map_resolver(self) -> MapResolver:
        shared_maps = self.resources.shared_maps
        if not shared_maps or self.map_config is None:
            return no_map()
        return shared_map(shared_maps, utils.extract_fn(self.map_config.extraction))

    def _recording_ids(self) -> list[int]:
        """Return sorted recording identifiers discovered from metadata files."""
        recording_ids: list[int] = []
        for recording_meta in sorted(self.root.glob("*_recordingMeta.csv")):
            prefix, _, _ = recording_meta.stem.partition("_")
            recording_ids.append(int(prefix))
        return recording_ids


_META_SCHEMA: pl.Schema = pl.Schema({"trackId": pl.Int32, "class": pl.Utf8})

_TRACK_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int32,
    "trackId": pl.Int32,
    "xCenter": pl.Float64,
    "yCenter": pl.Float64,
    "xVelocity": pl.Float64,
    "yVelocity": pl.Float64,
    "xAcceleration": pl.Float64,
    "yAcceleration": pl.Float64,
})
