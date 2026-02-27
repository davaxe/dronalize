from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.common.trajectory.basic import yaw_from_vel_expr
from dronalize.common.trajectory.process import prepare_agent_trajectories
from dronalize.core.datatypes import map_context as mc
from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.protocols.loader import BaseSceneLoader, LoaderConfig, Source

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class SindLoader(BaseSceneLoader[str, pl.LazyFrame]):
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
        super().__init__(loader_config, enforce_schema=True)
        self._data_dir = data_dir
        if self.loader_config.scene_filtering is not None and filter_parked_vehicles:
            # Will remove all agents with an average speed less than 0.1
            self.loader_config.scene_filtering.filter_slow_agents = 0.1

    @override
    def sources(self) -> Iterable[Source[str, pl.LazyFrame]]:
        for subdir in self._data_dir.iterdir():
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
                })
                .alias("agent_category"),
                pl.lit(None).alias("yaw"),
                *("x", "y", "vx", "vy", "ax", "ay"),
            )

            map_location = self._resolve_map(subdir.name)
            yield Source(
                identifier=subdir.name,
                inner=pl.concat([vehicle_df, pedestrian_df]),
                map_context=mc.Explicit(map=str(map_location)),
            )

    @override
    def num_sources(self) -> int | None:
        return sum(1 for _ in self._data_dir.iterdir())

    @override
    def load_raw(
        self, source: Source[str, pl.LazyFrame]
    ) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
        for df in prepare_agent_trajectories(source.inner, self.loader_config):
            yield df, source.map_context or mc.NoMap()

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns(
            pl
            .when(pl.col("yaw").is_null())
            .then(yaw_from_vel_expr())
            .otherwise(pl.col("yaw"))
            .alias("yaw")
        )

    @override
    def default_config(self) -> LoaderConfig:
        return LoaderConfig(20, 50, 0.1).window_parameters(25).scene_filtering_parameters()

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
