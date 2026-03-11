from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.config.loader import LoaderConfig
from dronalize.core.base import BaseSceneLoader
from dronalize.core.categories import AgentCategory
from dronalize.core.loader import IngestOutput, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable

# TODO: Add support for filtering parked vehicles.


class SindLoader(BaseSceneLoader[Path]):
    """Loader for the SIND dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
    ) -> None:
        """Initialize the SindLoader.

        Parameters
        ----------
        data_root : Path or str
            Path to the directory containing the SIND dataset.
        loader_config : LoaderConfig, optional
            Loader configuration override.

        """
        super().__init__(loader_config, enforce_schema=True)
        self._data_dir = self._normalize_data_root(data_root)

    @override
    def all_sources(self) -> Iterable[Source[Path]]:
        for subdir in self._data_dir.iterdir():
            map_location = self._resolve_map(subdir.name)
            yield Source(
                identifier=subdir.name,
                inner=subdir,
                map_key=str(map_location),
            )

    @override
    def ingest(self, source: Source[Path]) -> Iterable[IngestOutput]:
        subdir = source.inner
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
            })
            .alias("agent_category"),
            pl.col("heading_rad").alias("yaw"),
            *("x", "y", "vx", "vy", "ax", "ay"),
        )
        pedestrian_df = pl.scan_csv(
            pedestrian_data_path,
            schema_overrides=_PEDESTRIAN_SCHEMA,
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
            })
            .alias("agent_category"),
            pl.lit(None).alias("yaw"),
            *("x", "y", "vx", "vy", "ax", "ay"),
        )

        yield pl.concat([vehicle_df, pedestrian_df]), source.map_key

    @override
    def num_sources(self) -> int | None:
        if not self._data_dir.is_dir():
            return 0
        return sum(1 for _ in self._data_dir.iterdir())

    @override
    def pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .compose(
                trajectory_pipeline(self.loader_config, derivative_rename=self.derivative_names())
            )
            .then(tr.yaw_from_vel(only_null=True))
        )

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=20, output_len=50, sample_time=0.1)
            .with_window(25)
            .with_filtering(require_frames=[19])
        )

    @staticmethod
    def _resolve_map(path_name: str) -> str:
        if path_name.startswith("changchun"):
            return "Changchun_Pudong.osm"
        if path_name.startswith("xian"):
            return "Xi'an_Shanglin.osm"
        if "NR" in path_name:
            return "NR_ll2.osm"
        return "map_relink_law_save.osm"


_VEHICLE_SCHEMA = pl.Schema({
    "track_id": pl.Int32,
    "frame_id": pl.Int32,
    "agent_type": pl.Utf8,
    "heading_rad": pl.Float32,
    "x": pl.Float32,
    "y": pl.Float32,
    "vx": pl.Float32,
    "vy": pl.Float32,
    "ax": pl.Float32,
    "ay": pl.Float32,
})

_PEDESTRIAN_SCHEMA = pl.Schema({
    "track_id": pl.Utf8,
    "frame_id": pl.Int32,
    "agent_type": pl.Utf8,
    "x": pl.Float32,
    "y": pl.Float32,
    "vx": pl.Float32,
    "vy": pl.Float32,
    "ax": pl.Float32,
    "ay": pl.Float32,
})
