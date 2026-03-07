from __future__ import annotations

from dataclasses import replace as _replace
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.protocols.loader import BaseSceneLoader, IngestOutput, LoaderConfig, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable


class SindLoader(BaseSceneLoader[str, Path]):
    """Loader for the SIND dataset."""

    def __init__(
        self,
        data_dir: Path,
        loader_config: LoaderConfig | None = None,
        *,
        filter_parked_vehicles: bool = False,
    ) -> None:
        """Initialize the SindLoader.

        Parameters
        ----------
        data_dir : Path
            The directory containing the SIND dataset (i.e. the data directory).
        loader_config : LoaderConfig, optional
            Overrides for the default loader configuration.
        filter_parked_vehicles : bool, optional
            Whether to filter out parked vehicles based on their speed.
            If True, all agents with an average speed less than 0.1 will be
            removed. This will override any existing setting for the
            `filter_slow_agents` parameter in the `scene_filtering`
            configuration.

        """
        if filter_parked_vehicles:
            resolved = loader_config or type(self).default_config()
            filtering = resolved.scene_filtering
            if filtering is not None:
                loader_config = _replace(
                    resolved,
                    scene_filtering=_replace(filtering, filter_slow_agents=0.1),
                )
            else:
                loader_config = resolved.with_filtering(
                    require_frames=[resolved.input_len - 1],
                    filter_slow_agents=0.1 * 0.1,  # 0.1 m/s,
                )

        super().__init__(loader_config, enforce_schema=True)
        self._data_dir = data_dir

    @override
    def all_sources(self) -> Iterable[Source[str, Path]]:
        for subdir in self._data_dir.iterdir():
            map_location = self._resolve_map(subdir.name)
            yield Source(
                identifier=subdir.name,
                inner=subdir,
                map_key=str(map_location),
            )

    @override
    def ingest(self, source: Source[str, Path]) -> Iterable[IngestOutput]:
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
        return LoaderConfig(20, 50, 0.1).with_window(25).with_filtering(require_frames=[19])

    @staticmethod
    def _resolve_map(path_name: str) -> str:
        if path_name.startswith("changchun"):
            return "Changchun_Pudong.osm"
        if path_name.startswith("xian"):
            return "Xi'an_Shanglin.osm"
        if "NR" in path_name:
            return "NR_ll2.osm"
        return "map_relink_law_save.osm"


_VEHICLE_SCHEMA = {
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
}

_PEDESTRIAN_SCHEMA = {
    "track_id": pl.Utf8,
    "frame_id": pl.Int32,
    "agent_type": pl.Utf8,
    "x": pl.Float32,
    "y": pl.Float32,
    "vx": pl.Float32,
    "vy": pl.Float32,
    "ax": pl.Float32,
    "ay": pl.Float32,
}
