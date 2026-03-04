from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.datatypes.loader_config import LoaderConfig
from dronalize.core.protocols.loader import BaseSceneLoader, IngestOutput, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable


class XLevelDataLoader(BaseSceneLoader[int, Path]):
    """Common trajectory data loader for X-level datasets.

    This class is meant as a base class for the datasets, since some of the
    processing is slightly different for highway datasets (e.g., highD) compared
    to urban datasets (e.g., rounD, inD).

    With no changes this supports: rounD, inD, exiD, uniD, and sinD.
    """

    def __init__(
        self,
        data_dir: Path,
        loader_config: LoaderConfig | None = None,
    ) -> None:
        """Initialize the trajectory data loader for a X-level dataset (e.g., rounD, inD).

        Parameters
        ----------
        data_dir : Path
            Path to the directory containing the .csv data files.
        loader_config : LoaderConfig, optional
            Processor configuration. If None, default configuration will be used.

        """
        super().__init__(loader_config=loader_config, enforce_schema=False)
        self._data_dir = data_dir
        self._rebalance_ratio = None

    @staticmethod
    def meta_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the metadata CSV."""
        return [
            pl.col("trackId").alias("id"),
            pl
            .col("class")
            .replace_strict({
                "car": AgentCategory.CAR.value,
                "truck": AgentCategory.TRUCK.value,
                "bus": AgentCategory.BUS.value,
                "trailer": AgentCategory.TRAILER.value,
                "motorcycle": AgentCategory.MOTORCYCLE.value,
                "bicycle": AgentCategory.BICYCLE.value,
                "pedestrian": AgentCategory.PEDESTRIAN.value,
                "van": AgentCategory.VAN.value,
            })
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
    def meta_schema() -> pl.Schema:
        """Define the schema for the metadata CSV."""
        return _META_SCHEMA

    @staticmethod
    def track_schema() -> pl.Schema:
        """Define the schema for the track CSV."""
        return _TRACK_SCHEMA

    @override
    def all_sources(self) -> Iterable[Source[int, Path]]:
        num_files: int = sum(1 for p in self._data_dir.iterdir() if p.is_file())
        for i in range(1, num_files // 4):
            recording_meta = self._data_dir / f"{i:0>2}_recordingMeta.csv"
            recording_meta_data = pl.read_csv(recording_meta)
            location_id = recording_meta_data.select(pl.col("locationId")).item()
            columns = recording_meta_data.columns

            utm_x0: float | None = None
            utm_y0: float | None = None
            if "xUtmOrigin" in columns and "yUtmOrigin" in columns:
                utm_x0 = recording_meta_data.select(pl.col("xUtmOrigin")).item()
                utm_y0 = recording_meta_data.select(pl.col("yUtmOrigin")).item()

            metadata: dict[str, float] = {}
            if utm_x0 is not None and utm_y0 is not None:
                metadata["utm_x0"] = utm_x0
                metadata["utm_y0"] = utm_y0

            yield Source(
                identifier=i,
                inner=self._data_dir,
                map_key=str(location_id),
                metadata=metadata,
            )

    @override
    def ingest(self, source: Source[int, Path]) -> Iterable[IngestOutput]:
        tracks = source.inner / f"{source.identifier:0>2}_tracks.csv"
        meta = source.inner / f"{source.identifier:0>2}_tracksMeta.csv"
        meta_df = pl.scan_csv(meta, schema_overrides=self.meta_schema()).select(
            *self.meta_data_select(),
        )
        tracks_df = pl.scan_csv(tracks, schema_overrides=self.track_schema()).select(
            *self.track_data_select(),
        )
        combined = tracks_df.join(meta_df, left_on="id", right_on="id")
        yield combined, None

    @override
    def num_sources(self) -> int | None:
        num_files: int = sum(1 for p in self._data_dir.iterdir() if p.is_file())
        return num_files // 4 - 1

    @override
    def pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .then_if_present(
                tr.rebalance,
                arg=self._rebalance_ratio,
            )
            .compose(
                trajectory_pipeline(self.loader_config, derivative_rename=self.derivative_names())
            )
            .then(tr.yaw_from_vel())
        )

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=50, output_len=125, sample_time=0.04)
            .with_resampling(2, 5)
            .with_window(25)
            .with_filtering(
                require_frames=[49],
                filter_agent_category=[AgentCategory.TRAILER],
            )
        )


_META_SCHEMA: pl.Schema = pl.Schema({
    "trackId": pl.Int32,
    "class": pl.Utf8,
})

_TRACK_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int32,
    "trackId": pl.Int32,
    "xCenter": pl.Float32,
    "yCenter": pl.Float32,
    "xVelocity": pl.Float32,
    "yVelocity": pl.Float32,
    "xAcceleration": pl.Float32,
    "yAcceleration": pl.Float32,
})
