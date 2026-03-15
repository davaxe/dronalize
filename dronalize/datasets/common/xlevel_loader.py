from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.common import utils
from dronalize.loading import BaseSceneLoader
from dronalize.loading.loader import IngestOutput, Source
from dronalize.maps.resolver import MapResolver, no_map, shared_map
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable


class XLevelDataLoader(BaseSceneLoader[Path]):
    """Common trajectory data loader for X-level datasets.

    This class is meant as a base class for the datasets, since some of the
    processing is slightly different for highway datasets (e.g., highD) compared
    to urban datasets (e.g., rounD, inD).

    With no changes this supports: rounD, inD, exiD, uniD, and sinD.
    """

    def __init__(
        self,
        data_dir: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        """Initialize the loader for an X-level dataset (e.g., rounD, inD).

        Parameters
        ----------
        data_dir : Path or str
            Path to the directory containing the .csv data files.
        loader_config : LoaderConfig, optional
            Loader configuration. If None, the default configuration is used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. X-level datasets do not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(loader_config=loader_config, map_config=map_config, splits=splits)
        self._data_dir: Path = self._normalize_data_root(data_dir)
        self._rebalance_ratio: float | None = None

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
    def discover_sources(self) -> Iterable[Source[Path]]:
        for recording_id in self._recording_ids():
            recording_meta = self._data_dir / f"{recording_id:0>2}_recordingMeta.csv"
            recording_meta_data = pl.read_csv(recording_meta)
            location_id = self.location_id_select(recording_meta_data, recording_meta)
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
                identifier=recording_id,
                inner=self._data_dir,
                map_key=str(location_id),
                metadata=metadata,
            )

    @override
    def ingest(self, source: Source[Path]) -> Iterable[IngestOutput]:
        tracks = source.inner / f"{source.identifier:0>2}_tracks.csv"
        meta = source.inner / f"{source.identifier:0>2}_tracksMeta.csv"
        meta_df = pl.scan_csv(meta, schema_overrides=self.meta_schema()).select(
            *self.meta_data_select(),
        )
        tracks_df = pl.scan_csv(tracks, schema_overrides=self.track_schema()).select(
            *self.track_data_select(),
        )
        combined = tracks_df.join(meta_df, left_on="id", right_on="id")
        if "utm_x0" in source.metadata and "utm_y0" in source.metadata:
            combined = combined.with_columns(
                (pl.col("x") + source.metadata["utm_x0"]).alias("x"),
                (pl.col("y") + source.metadata["utm_y0"]).alias("y"),
            )

        yield combined, source.map_key

    @override
    def num_sources(self) -> int | None:
        return len(self._recording_ids())

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

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.no_extraction()

    @override
    def map_resolver(self) -> MapResolver:
        if self._shared_memory_name is None:
            return no_map()
        return shared_map(self._shared_memory_name, utils.extract_fn(self.map_config.extraction))

    def _recording_ids(self) -> list[int]:
        """Return sorted recording identifiers discovered from metadata files."""
        recording_ids: list[int] = []
        for recording_meta in sorted(self._data_dir.glob("*_recordingMeta.csv")):
            prefix, _, _ = recording_meta.stem.partition("_")
            recording_ids.append(int(prefix))
        return recording_ids


_META_SCHEMA: pl.Schema = pl.Schema({
    "trackId": pl.Int32,
    "class": pl.Utf8,
})

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
