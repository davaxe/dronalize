from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.ad4che.map.builder import AD4CHEMapBuilder
from dronalize.datasets.common import utils
from dronalize.datasets.common.xlevel_loader import XLevelDataLoader
from dronalize.loading import Source
from dronalize.scene import POSITIONS_VELOCITY_ACCELERATION_V1

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.maps import MapResolver
    from dronalize.maps.graph import MapGraph
    from dronalize.scene import Scene, SceneSchema


class AD4CHELoader(XLevelDataLoader):
    """Loader for the AD4CHE dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        *,
        lane_change_ratio: float | None = 1.0,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        """Initialize the trajectory data loader for the AD4CHE dataset.

        It is possible to rebalance the dataset by adjusting the number of lane
        changing agents compared to non-lane changing agents. This can be done
        by setting the `lane_change_ratio` parameter. For example, a ratio of
        0.5 would result in half as many lane changing agents as non-lane
        changing agents. Typically highway datasets are heavily imbalanced
        towards non-lane changing agents, which means that a high ratio con
        result in way less total data.

        Parameters
        ----------
        data_root : Path or str
            Path to the directory containing the .csv data files.
        loader_config : LoaderConfig, optional
            Loader configuration. If None, the default configuration is used.
        lane_change_ratio : float, optional
            Ratio to rebalance lane changing vs non-lane changing agents.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            Path(data_root) / "AD4CHE_Data_V1.0",
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
        )
        # Update internal state to enable rebalancing of lane changing vs non-lane changing agents
        self._rebalance_ratio: float | None = lane_change_ratio

    @override
    def discover_sources(self) -> Iterable[Source[Path]]:
        for recording_id, subdir in self._recordings():
            yield Source(
                identifier=recording_id,
                inner=subdir,
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
            .with_resampling(1, 3)
            .with_filtering(require_frames=[59])
            .with_window(45)
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.auto_extraction(padding_factor=1.15)

    @override
    def map_resolver(self) -> MapResolver:
        def _resolver(scene: Scene) -> MapGraph | None:
            if scene.map_key is None:
                return None
            path = self._data_dir / scene.map_key
            map_graph = AD4CHEMapBuilder(path).build(
                self.map_config.min_distance, self.map_config.interp_distance
            )
            return utils.extract_based_on_scene(map_graph, scene, self.map_config.extraction)

        return _resolver

    def _recordings(self) -> Iterable[tuple[int, Path]]:
        """Yield discovered recording identifiers with their directories and metadata files."""
        for subdir in sorted(path for path in self._data_dir.iterdir() if path.is_dir()):
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
