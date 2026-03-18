from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.common import utils
from dronalize.exceptions import SplitNotSupportedError
from dronalize.loading import BaseSceneLoader
from dronalize.loading.loader import IngestOutput, Source
from dronalize.maps import no_map
from dronalize.maps.resolver import MapResolver, shared_map
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline
from dronalize.scene import POSITIONS_ONLY_V1

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.scene import SceneSchema


class Argoverse1Loader(BaseSceneLoader[list[Path]]):
    """Loader for Argoverse 1 trajectory data stored in CSV format."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        *,
        file_batch_size: int | None = 100,
    ) -> None:
        """Initialize the dataset loader.

        Parameters
        ----------
        data_root : Path or str
            Path to the root directory of the Argoverse 1 dataset.
            This directory should contain the `forecasting_train_v1.1/`,
            `forecasting_val_v1.1/`, and `forecasting_test_v1.1/`
            subdirectories.
        file_batch_size : int, optional
            Number of files to read in each batch. If None, all files will be
            read at once. Higher batch size may lead to faster processing at
            diminishing returns, but also higher memory usage. `None` is not
            recommended for large amounts of data.
        loader_config : LoaderConfig, optional
            Loader configuration override. If None, the default configuration
            will be used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. Can contain one or more predefined splits,
            or `None` to process all sources.

        """
        super().__init__(loader_config=loader_config, map_config=map_config, splits=splits)
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
        if split is DatasetSplit.TEST:
            return self._sources_from_dir(self._test_dir)
        raise SplitNotSupportedError(type(self).__name__, split)

    @override
    def ingest(self, source: Source[list[Path]]) -> Iterable[IngestOutput]:
        batch_lf = (
            pl
            .scan_csv(source.inner, include_file_paths="file_id", schema=_SCHEMA)
            .with_columns(
                pl
                .when(pl.col("OBJECT_TYPE") == "AV")
                .then(AgentCategory.CAR)
                .otherwise(AgentCategory.UNKNOWN)
                .alias("agent_category"),
                pl.col("TRACK_ID").rank(method="dense").sub(1).cast(pl.Int64).alias("id"),
                pl.col("TIMESTAMP").rank(method="dense").sub(1).cast(pl.Int64).alias("frame"),
                pl.col("file_id").cast(pl.Categorical).to_physical(),
            )
            .drop("OBJECT_TYPE", "TRACK_ID", "TIMESTAMP")
            .rename({"X": "x", "Y": "y", "CITY_NAME": "map"})
        )

        for _, group in batch_lf.collect().group_by(["file_id"]):
            map_key = str(group["map"].first())
            yield group.lazy().drop("file_id"), map_key

    @override
    def num_sources(self) -> int | None:
        splits = (
            self.splits
            if self.splits is not None
            else self.predefined_splits()
        )
        return sum(self._count_sources_for_split(split) for split in splits)

    @override
    def pipeline(self) -> Pipeline:
        return Pipeline().compose(trajectory_pipeline(self.loader_config))

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_ONLY_V1

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(input_len=20, output_len=30, sample_time=0.1).with_filtering(
            require_frames=[19]
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.auto_extraction()

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
            yield Source(identifier=start, inner=batch_files)

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
        if split is DatasetSplit.TEST:
            return self._count_sources(self._test_dir)
        raise SplitNotSupportedError(type(self).__name__, split)


_SCHEMA: pl.Schema = pl.Schema({
    "TIMESTAMP": pl.Float64,
    "TRACK_ID": pl.String,
    "OBJECT_TYPE": pl.Categorical(pl.Categories("AV", "OTHERS")),
    "X": pl.Float64,
    "Y": pl.Float64,
    "CITY_NAME": pl.String,
})
