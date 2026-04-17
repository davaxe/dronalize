"""Loader implementation for the A43 dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory
from dronalize.core.scene import POSITIONS_VELOCITY_ACCELERATION, Scene
from dronalize.datasets.a43.maps.builder import A43MapBuilder
from dronalize.datasets.shared import utils
from dronalize.processing.loading.base import BaseSceneLoader, LoaderSplitCapabilities
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.processing.maps.resolver import MapResolver

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.maps.resolver import MapResolver
    from dronalize.processing.models import LoaderRequest


class A43Loader(BaseSceneLoader):
    """Scene loader for the A43 dataset."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_block_split=True
    )

    def __init__(self, *, data_root: Path | str, request: LoaderRequest) -> None:
        """Initialize the A43 dataset loader.

        Parameters
        ----------
        data_root : Path or str
            Root directory containing the extracted A43 CSV files.

        """
        super().__init__(data_root=data_root, request=request)

    @override
    def discover_sources(self) -> Iterable[Source[Path]]:
        for i, csv_file in enumerate(self.root.glob("*.csv")):
            yield Source(identifier=i, data=csv_file, map_key=csv_file.stem)

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        dt = 0.1
        eps = 1e-9
        yield LoadedSourceData(
            pl
            .scan_csv(source.data)
            .with_columns(t0=pl.col("tseconds").min())
            .with_columns(
                frame=(
                    (((pl.col("tseconds") - pl.col("t0")) / dt) + 0.5 + eps).floor().cast(pl.Int64)
                )
            )
            .select(
                pl.col("ID").alias("id"),
                pl.col("frame"),
                *[pl.col(c) for c in ("x", "y", "vy", "vx", "ax", "ay")],
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
            )
        )

    @override
    def num_sources(self) -> int | None:
        return sum(1 for f in self.root.rglob("*.csv") if f.is_file())

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_ACCELERATION

    @override
    def map_resolver(self) -> MapResolver:
        def _resolver(scene: Scene) -> MapGraph | None:
            if scene.map_key is None or self.map_config is None:
                return None

            min_x = scene.frame.select(pl.col("x")).min().item()
            max_x = scene.frame.select(pl.col("x")).max().item()
            builder = A43MapBuilder(scene.map_key, min_x, max_x)
            map_graph = builder.build(self.map_config.min_distance, self.map_config.interp_distance)
            return utils.extract_based_on_scene(map_graph, scene, self.map_config.extraction)

        return _resolver
