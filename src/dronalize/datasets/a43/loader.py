from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.a43.map.builder import A43MapBuilder
from dronalize.datasets.common import utils
from dronalize.loading.base import BaseSceneLoader, BaseSceneLoaderConfig
from dronalize.loading.loader import IngestedData, Source
from dronalize.maps.resolver import MapResolver
from dronalize.scene import POSITIONS_VELOCITY_ACCELERATION_V1, Scene

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.config.split import (
        SplitRequest,
    )
    from dronalize.maps.graph import MapGraph
    from dronalize.maps.resolver import MapResolver
    from dronalize.scene import SceneSchema


class A43Loader(BaseSceneLoader[Path]):
    """Scene loader for the A43 dataset."""

    config: ClassVar[BaseSceneLoaderConfig] = BaseSceneLoaderConfig(
        block_split_enabled=True,
    )

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitRequest | None = None,
    ) -> None:
        """Initialize the A43 dataset loader.

        Parameters
        ----------
        data_root : Path or str
            Root directory containing the extracted A43 CSV files.
        loader_config : LoaderConfig, optional
            Loader configuration override.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )
        self._data_root: Path = Path(data_root)

    @override
    def discover_sources(self) -> Iterable[Source[Path]]:
        for i, csv_file in enumerate(self._data_root.glob("*.csv")):
            yield Source(identifier=i, data=csv_file, map_key=csv_file.stem)

    @override
    def ingest(self, source: Source[Path]) -> Iterable[IngestedData]:
        yield IngestedData(
            pl.scan_csv(source.data).select(
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
        )

    @override
    def num_sources(self) -> int | None:
        return sum(1 for f in self._data_root.rglob("*.csv") if f.is_file())

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_VELOCITY_ACCELERATION_V1

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
