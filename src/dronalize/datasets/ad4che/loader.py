from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_VELOCITY_ACCELERATION_V1
from dronalize.datasets.ad4che.maps.builder import AD4CHEMapBuilder
from dronalize.datasets.shared import utils
from dronalize.datasets.shared.levelx_loader import LevelXDataLoader
from dronalize.processing.filters import Filter, RequireAgentFrames
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.ingest.loader import Source
from dronalize.processing.maps.config import MapConfig
from dronalize.processing.pipeline.functional.resample import ResampleSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.maps.graph import MapGraph
    from dronalize.core.scene import Scene, SceneSchema
    from dronalize.processing.ingest.splits import SplitRequest
    from dronalize.processing.maps.resolver import MapResolver


class AD4CHELoader(LevelXDataLoader):
    """Loader for the AD4CHE dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitRequest | None = None,
    ) -> None:
        """Initialize the trajectory data loader for the AD4CHE dataset.

        Parameters
        ----------
        data_root : Path or str
            Path to the directory containing the .csv data files.
        loader_config : LoaderConfig, optional
            Loader configuration. If None, the default configuration is used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            Path(data_root) / "AD4CHE_Data_V1.0",
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )

    @override
    def discover_sources(self) -> Iterable[Source[Path]]:
        for recording_id, subdir in self._recordings():
            yield Source(
                identifier=recording_id,
                data=subdir,
                map_key=f"{subdir.name}/{recording_id:02d}_laneWidthColorAndID.png",
            )

    @override
    def num_sources(self) -> int | None:
        return sum(1 for _ in self._recordings())

    @staticmethod
    @override
    def meta_data_select() -> list[pl.Expr]:
        """Select the relevant columns from the metadata CSV."""
        return [
            pl.col("id"),
            pl.col("numLaneChanges").alias("lane_changes"),
            pl
            .col("class")
            .replace_strict({
                "car": AgentCategory.CAR.value,
                "truck": AgentCategory.TRUCK.value,
                "bus": AgentCategory.BUS.value,
            })
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
        ]

    @staticmethod
    @override
    def meta_schema() -> pl.Schema:
        """Define the schema for the metadata CSV."""
        return _META_SCHEMA

    @staticmethod
    @override
    def track_schema() -> pl.Schema:
        """Define the schema for the track CSV."""
        return _TRACK_SCHEMA

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_VELOCITY_ACCELERATION_V1

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=60, output_len=150, sample_time=1 / 30)
            .with_resampling(ResampleSpec(up=1, down=3))
            .with_filters(Filter.define(filter_rules=[RequireAgentFrames.define(frames=[59])]))
            .with_window(45)
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.relevant_area_extraction(padding_factor=1.15)

    @override
    def map_resolver(self) -> MapResolver:
        def _resolver(scene: Scene) -> MapGraph | None:
            if scene.map_key is None:
                return None
            path = self._data_root / scene.map_key
            map_graph = AD4CHEMapBuilder(path).build(
                self.map_config.min_distance,
                self.map_config.interp_distance,
            )
            return utils.extract_based_on_scene(map_graph, scene, self.map_config.extraction)

        return _resolver

    def _recordings(self) -> Iterable[tuple[int, Path]]:
        """Yield discovered recording identifiers with their directories and metadata files."""
        for subdir in sorted(path for path in self._data_root.iterdir() if path.is_dir()):
            # subdir is on format DJI_XXXX
            number_str = subdir.name.split("_")[-1]
            yield int(number_str), subdir


_META_SCHEMA: pl.Schema = pl.Schema({
    "id": pl.Int32,
    "numLaneChanges": pl.Int8,
})

_TRACK_SCHEMA: pl.Schema = pl.Schema({
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
