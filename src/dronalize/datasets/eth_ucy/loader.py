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
        for data_file in self._source_files_for(split):
            yield DatasetSource(
                identifier=self._source_identifier(data_file=data_file, split=split),
                payload=data_file,
            )

    @override
    def load_source(self, source: DatasetSource[Path]) -> Iterable[LoadedSourceFrame]:
        yield LoadedSourceFrame(
            pl.scan_csv(
                source.payload,
                has_header=False,
                separator="\t",
                new_columns=["frame", "id", "x", "y"],
                schema_overrides={
                    "frame": pl.Float64,
                    "id": pl.Float64,
                    "x": pl.Float64,
                    "y": pl.Float64,
                },
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
        return sum(1 for _ in self._source_files_for(split))

    def _source_files_for(self, split: DatasetSplit) -> tuple[Path, ...]:
        split_dirs = sorted(
            path
            for path in self.root.rglob(split.value)
            if path.is_dir() and path.name == split.value
        )
        files = {
            data_file
            for split_dir in split_dirs
            for data_file in split_dir.rglob("*")
            if data_file.is_file()
        }
        return tuple(sorted(files))

    def _source_identifier(self, *, data_file: Path, split: DatasetSplit) -> str:
        direct_split_dir = self.root / split.value
        if data_file.parent == direct_split_dir:
            return data_file.name
        return data_file.relative_to(self.root).as_posix()
