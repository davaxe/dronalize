from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.a43.map.builder import A43MapBuilder
from dronalize.datasets.common import utils
from dronalize.loading import BaseSceneLoader
from dronalize.loading.loader import Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline
from dronalize.scene import Scene

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.loading.loader import IngestOutput
    from dronalize.maps import MapResolver
    from dronalize.maps.graph import MapGraph
    from dronalize.scene import Scene


class A43Loader(BaseSceneLoader[Path]):
    """Scene loader for the A43 dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        """Initialize the A43 dataset loader.

        Parameters
        ----------
        data_root : Path or str
            Path to root of the A43 dataset, data files.
        loader_config : LoaderConfig, optional
            Loader configuration. If None, the default configuration is used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(loader_config=loader_config, map_config=map_config, splits=splits)
        self._data_dir: Path = self._normalize_data_root(data_root)

    @override
    def discover_sources(self) -> Iterable[Source[Path]]:
        for i, csv_file in enumerate(self._data_dir.glob("*.csv")):
            yield Source(identifier=i, inner=csv_file)

    @override
    def ingest(self, source: Source[Path]) -> Iterable[IngestOutput]:
        yield (
            pl.scan_csv(source.inner).select(
                pl.col("ID").alias("id"),
                pl.col("tseconds").round(1).rank("dense").sub(1).alias("frame").cast(pl.Int64),
                *("x", "y", "vy", "vx", "ax", "ay"),
                pl
                .col("VehicleCategory")
                .replace_strict({
                    "Motorcycle": AgentCategory.MOTORCYCLE,
                    "Passenger Car": AgentCategory.CAR,
                    "Semi-trailer truck": AgentCategory.TRUCK,
                    "Truck": AgentCategory.TRUCK,
                    "Van": AgentCategory.VAN,
                    "Bus": AgentCategory.BUS,
                })
                .alias("agent_category"),
            ),
            source.inner.stem,
        )

    @override
    def num_sources(self) -> int | None:
        return self._count_matching_files([self._data_dir], "*.csv")

    @override
    def pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .compose(
                trajectory_pipeline(self.loader_config, derivative_rename=self.derivative_names())
            )
            .then(tr.yaw_from_vel())
        )

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=20, output_len=50, sample_time=0.1)
            .with_window(25)
            .with_filtering(require_frames=[19])
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.no_extraction()

    @override
    def map_resolver(self) -> MapResolver:
        def _resolver(scene: Scene) -> MapGraph | None:
            if scene.map_key is None:
                return None

            min_x = scene.inner.select(pl.col("x")).min().item()
            max_x = scene.inner.select(pl.col("x")).max().item()
            builder = A43MapBuilder(scene.map_key, min_x, max_x)
            map_graph = builder.build(self.map_config.min_distance, self.map_config.interp_distance)
            return utils.extract_based_on_scene(map_graph, scene, self.map_config.extraction)

        return _resolver
