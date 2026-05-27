"""Loader implementation for the I-80 dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory
from dronalize.core.scene import POSITIONS_ONLY
from dronalize.datasets.shared import utils
from dronalize.processing.loading.base import SceneLoader
from dronalize.processing.loading.models import DatasetSource, LoadedSourceFrame
from dronalize.processing.maps import MapResolver, no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.core.scene import TrajectorySchema


class NGSimLoader(SceneLoader):
    """Scene loader for the I-80 dataset."""

    @override
    def iter_sources(self) -> Iterable[DatasetSource[Path]]:
        for i, csv_file in enumerate(sorted(self.root.rglob("trajectories*.csv"))):
            yield DatasetSource(identifier=i, payload=csv_file)

    @override
    def load_source(self, source: DatasetSource[Path]) -> Iterable[LoadedSourceFrame]:
        yield LoadedSourceFrame(
            pl.scan_csv(source.payload).select(
                pl.col("Vehicle_ID").alias("id"),
                pl.col("Frame_ID").alias("frame"),
                pl.col("Local_X").alias("x").mul(0.3048),
                pl.col("Local_Y").alias("y").mul(0.3048),
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
    def count_sources(self) -> int | None:
        return sum(1 for path in self.root.rglob("trajectories*.csv") if path.is_file())

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY

    @override
    def map_resolver(self) -> MapResolver:
        shared_maps = self.resources.shared_maps
        if not shared_maps or self.map_config is None:
            return no_map()
        return shared_map(shared_maps, utils.extract_fn(self.map_config.extraction))
