"""Loader implementations for LevelX-style datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory
from dronalize.core.scene import CANONICAL, POSITIONS_VELOCITY_ACCELERATION
from dronalize.datasets.levelx.maps import HighDMapBuilder
from dronalize.datasets.shared import utils
from dronalize.processing.loading.base import SceneLoader
from dronalize.processing.loading.models import DatasetSource, LoadedSourceFrame
from dronalize.processing.maps import MapResolver, no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import Scene, TrajectorySchema
    from dronalize.processing.loading.models import DatasetRunResources
    from dronalize.processing.models import LoaderPlan


@dataclass(slots=True, frozen=True)
class LevelXSourceData:
    path: Path
    x0: float | None = None
    y0: float | None = None


class LevelXDataLoader(SceneLoader[LevelXSourceData]):
    """Common recording CSV loader for LevelX-style datasets."""

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
        """Select the map key from the recording metadata CSV."""
        _ = path
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
    def iter_sources(self) -> Iterable[DatasetSource[LevelXSourceData]]:
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

            yield DatasetSource(
                identifier=recording_id,
                payload=LevelXSourceData(self.root, x0=utm_x0, y0=utm_y0),
                map_key=str(location_id),
            )

    @override
    def load_source(self, source: DatasetSource[LevelXSourceData]) -> Iterable[LoadedSourceFrame]:
        tracks = source.payload.path / f"{source.identifier:0>2}_tracks.csv"
        meta = source.payload.path / f"{source.identifier:0>2}_tracksMeta.csv"
        meta_df = pl.scan_csv(meta, schema_overrides=self.meta_schema()).select(
            *self.meta_data_select()
        )
        tracks_df = pl.scan_csv(tracks, schema_overrides=self.track_schema()).select(
            *self.track_data_select()
        )
        combined = tracks_df.join(meta_df, left_on="id", right_on="id")
        combined = combined.with_columns(
            (pl.col("x") + (source.payload.x0 or 0.0)).alias("x"),
            (pl.col("y") + (source.payload.y0 or 0.0)).alias("y"),
        )
        yield LoadedSourceFrame(combined)

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


class StandardLevelXLoader(LevelXDataLoader):
    """Loader for LevelX datasets that store recordings under a ``data`` directory."""

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderPlan,
        resources: DatasetRunResources | None = None,
    ) -> None:
        super().__init__(data_root=Path(data_root) / "data", request=request, resources=resources)


class ExiDLoader(StandardLevelXLoader):
    """Loader for the ExiD dataset."""

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
        return _EXID_TRACK_SCHEMA


class HighDLoader(LevelXDataLoader):
    """Loader for the highD dataset."""

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderPlan,
        resources: DatasetRunResources | None = None,
    ) -> None:
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
            .replace_strict(
                {"Car": AgentCategory.CAR.value, "Truck": AgentCategory.TRUCK.value},
                return_dtype=pl.Int32,
            )
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
        return _HIGHD_META_SCHEMA

    @staticmethod
    @override
    def track_schema() -> pl.Schema:
        """Define the schema for the track CSV."""
        return _HIGHD_TRACK_SCHEMA

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
            map_graph = builder.build(
                self.map_config.min_distance, self.map_config.interpolation_distance
            )
            return utils.extract_configured_map(map_graph, scene, self.map_config)

        return _resolver


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

_EXID_TRACK_SCHEMA: pl.Schema = pl.Schema({
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

_HIGHD_META_SCHEMA: pl.Schema = pl.Schema({"id": pl.Int32, "numLaneChanges": pl.Int8})

_HIGHD_TRACK_SCHEMA: pl.Schema = pl.Schema({
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
