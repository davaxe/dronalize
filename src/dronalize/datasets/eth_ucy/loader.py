"""Loader implementations for the ETH/UCY pedestrian datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_ONLY
from dronalize.processing.loading.base import BaseSceneLoader
from dronalize.processing.loading.loader import LoadedSourceData, Source

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.models import LoaderRequest


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


class _EthUcyLoader(BaseSceneLoader):
    """Loader for ETH/UCY pedestrian trajectory datasets."""

    def __init__(self, data_root: Path | str, request: LoaderRequest) -> None:
        super().__init__(data_root=data_root, request=request)

    def _sources_from_split(self, split_name: str) -> Iterable[Source[Path]]:
        data_dir = self.root / split_name
        for data_file in sorted(data_dir.iterdir()):
            yield Source(identifier=data_file.name, data=data_file)

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[Path]]:
        return self._sources_from_split(split.value)

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        yield LoadedSourceData(
            pl.scan_csv(
                source.data,
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
    def num_sources(self) -> int | None:
        return sum(
            self._count_sources_for_split(split) for split in self.native_splits or _NATIVE_SPLITS
        )

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        data_dir = self.root / split.value
        return sum(1 for _ in data_dir.iterdir()) if data_dir.is_dir() else 0


class HotelLoader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY hotel dataset."""


class EthLoader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY eth dataset."""


class UnivLoader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY univ dataset."""


class Zara1Loader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY zara1 dataset."""


class Zara2Loader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY zara2 dataset."""
