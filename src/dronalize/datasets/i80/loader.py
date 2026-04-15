"""Loader implementation for the I-80 dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory
from dronalize.core.scene import POSITIONS_ONLY
from dronalize.datasets.shared import utils
from dronalize.processing.loading.base import BaseSceneLoader, LoaderSplitCapabilities
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.processing.maps.resolver import MapResolver, no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.models import LoaderRequest


class I80Loader(BaseSceneLoader):
    """Scene loader for the I-80 dataset."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_block_split=True
    )

    def __init__(
        self,
        *,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        """Initialize the I-80 dataset loader."""
        super().__init__(data_root=data_root, request=request, resources=resources)

    @classmethod
    @override
    def unified_factory(
        cls,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> I80Loader:
        return cls(data_root=data_root, request=request, resources=resources)

    @override
    def discover_sources(self) -> Iterable[Source[Path]]:
        for i, csv_file in enumerate(sorted(self.root.rglob("trajectories*.csv"))):
            yield Source(identifier=i, data=csv_file)

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        yield LoadedSourceData(
            pl.scan_csv(source.data).select(
                pl.col("Vehicle_ID").alias("id"),
                pl.col("Frame_ID").alias("frame"),
                pl.col("Local_X").alias("x"),
                pl.col("Local_Y").alias("y"),
                pl
                .col("v_Class")
                .replace_strict({
                    1: AgentCategory.MOTORCYCLE,
                    2: AgentCategory.CAR,
                    3: AgentCategory.TRUCK,
                })
                .alias("agent_category"),
                pl.col("Lane_ID").alias("lane_id"),
            )
        )

    @override
    def num_sources(self) -> int | None:
        return sum(1 for path in self.root.rglob("trajectories*.csv") if path.is_file())

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY

    @override
    def map_resolver(self) -> MapResolver:
        shared_maps = self.resources.shared_maps
        if not isinstance(shared_maps, str) or self.map_config is None:
            return no_map()
        return shared_map(shared_maps, utils.extract_fn(self.map_config.extraction))
