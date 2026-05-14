"""Loader implementations for the ETH/UCY pedestrian datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_ONLY
from dronalize.processing.loading.base import SceneLoader
from dronalize.processing.loading.models import DatasetSource, LoadedSourceFrame

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.core.scene import TrajectorySchema


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


class EthUcyLoader(SceneLoader):
    """Loader for ETH/UCY pedestrian trajectory datasets."""

    @override
    def iter_sources_for(self, split: DatasetSplit) -> Iterable[DatasetSource[Path]]:
        data_dir = self.root / split.value
        for data_file in sorted(data_dir.iterdir()):
            yield DatasetSource(identifier=data_file.name, payload=data_file)

    @override
    def load_source(self, source: DatasetSource[Path]) -> Iterable[LoadedSourceFrame]:
        yield LoadedSourceFrame(
            pl.scan_csv(
                source.payload,
                has_header=False,
                separator="\t",
                new_columns=["frame", "id", "x", "y"],
                schema=pl.Schema({
                    "frame": pl.Int32,
                    "id": pl.Int32,
                    "x": pl.Float64,
                    "y": pl.Float64,
                }),
            ).with_columns(
                ((pl.col("frame") - pl.col("frame").min()) // 10).cast(pl.Int32),
                pl.col("id").cast(pl.Int32),
                agent_category=pl.lit(AgentCategory.PEDESTRIAN),
            )
        )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY

    @override
    def count_sources_for(self, split: DatasetSplit) -> int | None:
        data_dir = self.root / split.value
        return sum(1 for _ in data_dir.iterdir()) if data_dir.is_dir() else 0
