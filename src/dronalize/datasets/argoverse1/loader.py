from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_ONLY_V1
from dronalize.datasets.shared import utils
from dronalize.processing.filters import Filter
from dronalize.processing.filters.agent import RequireFrames
from dronalize.processing.ingest.base import BaseSceneLoader, LoaderSplitCapabilities
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.ingest.loader import IngestedData, MapBinding, Source
from dronalize.processing.maps.config import MapConfig
from dronalize.processing.maps.resolver import MapResolver, no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.scene import SceneSchema
    from dronalize.processing.ingest.splits import SplitRequest


class Argoverse1Loader(BaseSceneLoader[list[Path]]):
    """Loader for Argoverse 1 trajectory data stored in CSV format."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_scene_split=True
    )

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitRequest | None = None,
        *,
        file_batch_size: int | None = 10,
    ) -> None:
        """Initialize the Argoverse 1 loader.

        Parameters
        ----------
        data_root : Path or str
            Root directory of the Argoverse 1 forecasting dataset. It should
            contain `forecasting_train_v1.1/`, `forecasting_val_v1.1/`, and
            `forecasting_test_v1.1/`.
        file_batch_size : int, optional
            Number of files to read in each batch. If None, all files will be
            read at once. Larger batches can improve throughput at the cost of
            higher memory usage. `None` is usually only reasonable for small
            datasets.
        loader_config : LoaderConfig, optional
            Loader configuration override.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Optional selection of predefined dataset splits. `None` processes
            all available sources.
        """
        super().__init__(
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )
        self._data_root: Path = Path(data_root)
        self._batch_size: int | None = file_batch_size
        self._train_dir: Path = self._data_root / "forecasting_train_v1.1" / "train" / "data"
        self._val_dir: Path = self._data_root / "forecasting_val_v1.1" / "val" / "data"
        self._test_dir: Path = self._data_root / "forecasting_test_v1.1" / "test_obs" / "data"

    @classmethod
    @override
    def predefined_splits(cls) -> tuple[DatasetSplit, ...]:
        return (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[list[Path]]]:
        if split is DatasetSplit.TRAIN:
            return self._sources_from_dir(self._train_dir)
        if split is DatasetSplit.VAL:
            return self._sources_from_dir(self._val_dir)
        return self._sources_from_dir(self._test_dir)

    @override
    def ingest(self, source: Source[list[Path]]) -> Iterable[IngestedData]:
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
            map_key = str(group["map"].first())
            yield IngestedData(
                frame=group.drop("file_id").lazy(), map_binding=MapBinding(map_key=map_key)
            )

    @override
    def num_sources(self) -> int | None:
        splits = self.splits if self.splits is not None else self.predefined_splits()
        return sum(self._count_sources_for_split(split) for split in splits)

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_ONLY_V1

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(input_len=20, output_len=30, sample_time=0.1).with_filter(
            Filter.define(agent_rules=[RequireFrames.define(frames=[19])])
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.relevant_area_extraction()

    @override
    def map_resolver(self) -> MapResolver:
        if self._shared_memory_name is None:
            return no_map()
        return shared_map(self._shared_memory_name, utils.extract_fn(self.map_config.extraction))

    def _sources_from_dir(self, data_dir: Path) -> Iterable[Source[list[Path]]]:
        if not data_dir.is_dir():
            return
        files: list[Path] = sorted(data_dir.glob("*.csv"))
        batch_size: int = self._batch_size or len(files) or 1
        for start in range(0, len(files), batch_size):
            batch_files = files[start : start + batch_size]
            yield Source(identifier=start, data=batch_files)

    def _count_sources(self, data_dir: Path) -> int:
        if not data_dir.is_dir():
            return 0
        num_files = sum(1 for _ in data_dir.glob("*.csv"))
        if num_files == 0:
            return 0
        batch_size = self._batch_size or num_files
        batches, extra = divmod(num_files, batch_size)
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


if __name__ == "__main__":
    from dronalize.datasets.argoverse1 import DESCRIPTOR
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env

    root = resolve_dataset_root_from_env("argoverse1")
    _ = debug_descriptor(DESCRIPTOR, root)
