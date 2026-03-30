from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import CANONICAL_V1
from dronalize.datasets.shared import utils
from dronalize.processing.filters import Filter
from dronalize.processing.filters.agent import RequireFrames
from dronalize.processing.filters.cleanup import ExcludeCategories
from dronalize.processing.ingest.base import BaseSceneLoader, LoaderSplitCapabilities
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.ingest.loader import IngestedData, Source
from dronalize.processing.maps.config import MapConfig
from dronalize.processing.maps.resolver import MapResolver, no_map, shared_map
from dronalize.processing.pipeline.functional.resample import ResampleSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.scene import SceneSchema
    from dronalize.processing.ingest.splits import SplitRequest


class LevelXDataLoader(BaseSceneLoader[Path]):
    """Common trajectory data loader for X-level datasets.

    Each discovered source corresponds to one full recording. The source payload
    stays lightweight: it is just the shared dataset root plus a stable
    recording identifier.

    With no changes this supports: rounD, inD, exiD, uniD, and sinD.
    """

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_source_split=True
    )

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitRequest | None = None,
    ) -> None:
        """Initialize the loader for an X-level dataset (e.g., rounD, inD).

        Parameters
        ----------
        data_root : Path or str
            Root directory containing the extracted recording CSV files.
        loader_config : LoaderConfig, optional
            Loader configuration. If None, the default configuration is used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. X-level datasets do not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )
        self._data_root: Path = Path(data_root)

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
            .cast(pl.Int32())
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
            recording_meta = self._data_root / f"{recording_id:0>2}_recordingMeta.csv"
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
                data=self._data_root,
                map_key=str(location_id),
                metadata=metadata,
            )

    @override
    def ingest(self, source: Source[Path]) -> Iterable[IngestedData]:
        tracks = source.data / f"{source.identifier:0>2}_tracks.csv"
        meta = source.data / f"{source.identifier:0>2}_tracksMeta.csv"
        meta_df = pl.scan_csv(meta, schema_overrides=self.meta_schema()).select(
            *self.meta_data_select()
        )
        tracks_df = pl.scan_csv(tracks, schema_overrides=self.track_schema()).select(
            *self.track_data_select()
        )
        combined = tracks_df.join(meta_df, left_on="id", right_on="id")
        if "utm_x0" in source.metadata and "utm_y0" in source.metadata:
            combined = combined.with_columns(
                (pl.col("x") + source.metadata["utm_x0"]).alias("x"),
                (pl.col("y") + source.metadata["utm_y0"]).alias("y"),
            )
        yield IngestedData(combined)

    @override
    def num_sources(self) -> int | None:
        return len(self._recording_ids())

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return CANONICAL_V1

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=50, output_len=125, sample_time=0.04)
            .with_resampling(ResampleSpec(up=2, down=5))
            .with_window(25)
            .with_filter(
                Filter.define(
                    cleanup_rules=[ExcludeCategories.define(categories=[AgentCategory.TRAILER])],
                    agent_rules=[RequireFrames.define(frames=[49])],
                )
            )
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.full_map()

    @override
    def map_resolver(self) -> MapResolver:
        if self._shared_memory_name is None:
            return no_map()
        return shared_map(self._shared_memory_name, utils.extract_fn(self.map_config.extraction))

    def _recording_ids(self) -> list[int]:
        """Return sorted recording identifiers discovered from metadata files."""
        recording_ids: list[int] = []
        for recording_meta in sorted(self._data_root.glob("*_recordingMeta.csv")):
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
