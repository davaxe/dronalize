"""Loader implementation for the INTERACTION dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from pydantic import Field
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_VELOCITY_YAW
from dronalize.processing.loading.base import (
    BaseSceneLoader,
    LoaderOptions,
    LoaderSplitCapabilities,
)
from dronalize.processing.loading.loader import LoadedSourceData, Source

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.models import LoaderRequest


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


class InteractionLoaderOptions(LoaderOptions):
    """Dataset-owned config for the INTERACTION loader."""

    file_batch_size: int = Field(default=100, ge=1)


class InteractionLoader(BaseSceneLoader[list[Path], InteractionLoaderOptions]):
    """Loader for the INTERACTION dataset."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_scene_split=True
    )

    def __init__(
        self,
        *,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        """Initialize the INTERACTION loader."""
        super().__init__(data_root=data_root, request=request, resources=resources)

    @classmethod
    @override
    def loader_options_model(cls) -> type[InteractionLoaderOptions]:
        return InteractionLoaderOptions

    def _sources_from_dir(self, data_dir: Path) -> Iterable[Source[list[Path]]]:
        if not data_dir.is_dir():
            return
        csv_files = sorted(data_dir.rglob("*.csv"))
        for start in range(0, len(csv_files), self.loader_options.file_batch_size):
            yield Source(
                f"{data_dir.name}_b{start}",
                csv_files[start : start + self.dataset_config.file_batch_size],
            )

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[list[Path]]]:
        if split is DatasetSplit.TRAIN:
            yield from self._sources_from_dir(self.root / "train")
            return
        if split is DatasetSplit.VAL:
            yield from self._sources_from_dir(self.root / "val")
            return
        if split is DatasetSplit.TEST:
            yield from self._sources_from_dir(self.root / "test_multi-agent")
            yield from self._sources_from_dir(self.root / "test_conditional-multi-agent")

    @override
    def load_source(self, source: Source[list[Path]]) -> Iterable[LoadedSourceData]:
        data = (
            pl
            .scan_csv(source.data, include_file_paths="file_id", schema=_SCHEMA)
            .drop("track_to_predict", "interesting_agent", "width", "length", "timestamp_ms")
            .rename({"psi_rad": "yaw", "frame_id": "frame", "track_id": "id"})
            .with_columns(
                pl.col("file_id").cast(pl.Categorical).to_physical(),
                pl.col("case_id").cast(pl.UInt32),
                self._map_agent_category().alias("agent_category"),
            )
            .drop("agent_type")
        )

        for _, group in data.collect().group_by(["file_id", "case_id"]):
            yield LoadedSourceData(frame=group.lazy().drop("file_id", "case_id"))

    @override
    def num_sources(self) -> int | None:
        return sum(
            self._count_sources_for_split(split) for split in self.native_splits or _NATIVE_SPLITS
        )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_YAW

    @staticmethod
    def _map_agent_category() -> pl.Expr:
        return (
            pl
            .when(pl.col("agent_type") == "car")
            .then(AgentCategory.CAR.value)
            .when(pl.col("agent_type").is_in(["pedestrian/bicycle"]))
            .then(
                pl
                .when((pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt() < 2)
                .then(AgentCategory.PEDESTRIAN.value)
                .otherwise(AgentCategory.BICYCLE.value)
            )
            .otherwise(pl.col("agent_type"))
        )

    def _count_sources(self, data_dir: Path) -> int:
        if not data_dir.is_dir():
            return 0
        num_files = sum(1 for _ in data_dir.rglob("*.csv"))
        batches, extra = divmod(num_files, self.loader_options.file_batch_size)
        return batches + int(extra > 0)

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self.root / "train")
        if split is DatasetSplit.VAL:
            return self._count_sources(self.root / "val")
        return self._count_sources(self.root / "test_multi-agent") + self._count_sources(
            self.root / "test_conditional-multi-agent"
        )


_SCHEMA = pl.Schema({
    "case_id": pl.Float64,
    "track_id": pl.UInt32,
    "frame_id": pl.UInt32,
    "timestamp_ms": pl.Float64,
    "agent_type": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "psi_rad": pl.Float64,
    "length": pl.Float64,
    "width": pl.Float64,
    "track_to_predict": pl.Float64,
    "interesting_agent": pl.Float64,
})
