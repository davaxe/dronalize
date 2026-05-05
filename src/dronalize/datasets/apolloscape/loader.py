"""Loader implementation for the ApolloScape dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.errors import SplitNotSupportedError
from dronalize.core.scene import POSITIONS_YAW
from dronalize.processing.loading.base import BaseSceneLoader
from dronalize.processing.loading.loader import LoadedSourceData, Source

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.core.scene import TrajectorySchema


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL)


class ApolloScapeLoader(BaseSceneLoader):
    """Loader for ApolloScape prediction trajectories."""

    @staticmethod
    def _sources_from_dir(data_dir: Path) -> Iterable[Source[Path]]:
        if not data_dir.is_dir():
            return
        for data_file in sorted(data_dir.glob("*.txt")):
            yield Source(identifier=data_file.stem, data=data_file)

    @override
    def iter_sources_for(self, split: DatasetSplit) -> Iterable[Source[Path]]:
        if split is DatasetSplit.TRAIN:
            yield from self._sources_from_dir(self.root / "prediction_train")
            return
        if split is DatasetSplit.VAL:
            yield from self._sources_from_dir(self.root / "val_split")
            return
        raise SplitNotSupportedError(type(self).__name__, split)

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        yield LoadedSourceData(
            pl.scan_csv(source.data, has_header=False, schema=_DATA_SCHEMA, separator=" ").select(
                *("frame", "id", "x", "y", "yaw"),
                pl.col("agent_category").replace_strict({
                    1: AgentCategory.CAR.value,
                    2: AgentCategory.TRUCK.value,
                    3: AgentCategory.PEDESTRIAN.value,
                    4: AgentCategory.BICYCLE.value,
                    5: AgentCategory.UNKNOWN.value,
                }),
            )
        )

    @override
    def count_sources_for(self, split: DatasetSplit) -> int | None:
        if split is DatasetSplit.TRAIN:
            return sum(1 for _ in (self.root / "prediction_train").glob("*.txt"))
        if split is DatasetSplit.VAL:
            return sum(1 for _ in (self.root / "val_split").glob("*.txt"))
        raise SplitNotSupportedError(type(self).__name__, split)

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_YAW


_DATA_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int64,
    "id": pl.Int64,
    "agent_category": pl.Int64,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "length": pl.Float64,
    "width": pl.Float64,
    "height": pl.Float64,
    "yaw": pl.Float64,
})
