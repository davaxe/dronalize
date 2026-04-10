"""Loader implementation for the Argoverse 1 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from pydantic import Field
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_ONLY
from dronalize.datasets.shared import utils
from dronalize.processing.loading.base import (
    BaseSceneLoader,
    LoaderOptions,
    LoaderSplitCapabilities,
)
from dronalize.processing.loading.loader import LoadedSourceData, MapBinding, Source
from dronalize.processing.maps.resolver import MapResolver, no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.models import LoaderRequest


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


class Argoverse1LoaderOptions(LoaderOptions):
    """Dataset-owned config for the Argoverse 1 loader."""

    file_batch_size: int = Field(default=10, ge=1)


class Argoverse1Loader(BaseSceneLoader[list[Path], Argoverse1LoaderOptions]):
    """Loader for Argoverse 1 forecasting trajectories."""

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
        """Initialize the Argoverse 1 loader."""
        super().__init__(data_root=data_root, request=request, resources=resources)
        self._train_dir: Path = self.root / "forecasting_train_v1.1" / "train" / "data"
        self._val_dir: Path = self.root / "forecasting_val_v1.1" / "val" / "data"
        self._test_dir: Path = self.root / "forecasting_test_v1.1" / "test_obs" / "data"

    @classmethod
    @override
    def loader_options_model(cls) -> type[Argoverse1LoaderOptions]:
        return Argoverse1LoaderOptions

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[list[Path]]]:
        if split is DatasetSplit.TRAIN:
            return self._sources_from_dir(self._train_dir)
        if split is DatasetSplit.VAL:
            return self._sources_from_dir(self._val_dir)
        return self._sources_from_dir(self._test_dir)

    @override
    def load_source(self, source: Source[list[Path]]) -> Iterable[LoadedSourceData]:
        batch_lf = (
            pl
            .scan_csv(source.data, include_file_paths="file_id", schema=_SCHEMA)
            .with_columns(
                pl
                .when(pl.col("OBJECT_TYPE") == "AV")
                .then(AgentCategory.CAR)
                .otherwise(AgentCategory.UNKNOWN)
                .alias("agent_category"),
                pl
                .col("TRACK_ID")
                .rank(method="dense")
                .over("file_id")
                .sub(1)
                .cast(pl.Int64)
                .alias("id"),
                pl
                .col("TIMESTAMP")
                .rank(method="dense")
                .over("file_id")
                .sub(1)
                .cast(pl.Int64)
                .alias("frame"),
                pl.col("file_id").cast(pl.Categorical).to_physical(),
            )
            .drop("OBJECT_TYPE", "TRACK_ID", "TIMESTAMP")
            .rename({"X": "x", "Y": "y", "CITY_NAME": "map"})
        )

        for _, group in batch_lf.collect().group_by(["file_id"]):
            yield LoadedSourceData(
                frame=group.drop("file_id").lazy(),
                map_binding=MapBinding(map_key=str(group["map"].first())),
            )

    @override
    def num_sources(self) -> int | None:
        return sum(
            self._count_sources_for_split(split) for split in self.native_splits or _NATIVE_SPLITS
        )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY

    @override
    def map_resolver(self) -> MapResolver:
        shared_maps = self.resources.shared_maps
        if not isinstance(shared_maps, dict) or self.map_config is None:
            return no_map()
        return shared_map(shared_maps, utils.extract_fn(self.map_config.extraction))

    def _sources_from_dir(self, data_dir: Path) -> Iterable[Source[list[Path]]]:
        if not data_dir.is_dir():
            return
        files = sorted(data_dir.glob("*.csv"))
        for start in range(0, len(files), self.loader_options.file_batch_size):
            yield Source(
                identifier=start, data=files[start : start + self.loader_options.file_batch_size]
            )

    def _count_sources(self, data_dir: Path) -> int:
        if not data_dir.is_dir():
            return 0
        num_files = sum(1 for _ in data_dir.glob("*.csv"))
        batches, extra = divmod(num_files, self.loader_options.file_batch_size)
        return batches + int(extra > 0)

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self._train_dir)
        if split is DatasetSplit.VAL:
            return self._count_sources(self._val_dir)
        return self._count_sources(self._test_dir)


_SCHEMA: pl.Schema = pl.Schema({
    "TIMESTAMP": pl.Float64,
    "TRACK_ID": pl.String,
    "OBJECT_TYPE": pl.Categorical(pl.Categories("AV", "OTHERS")),
    "X": pl.Float64,
    "Y": pl.Float64,
    "CITY_NAME": pl.String,
})
