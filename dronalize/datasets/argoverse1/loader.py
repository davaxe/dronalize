from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.common import utils
from dronalize.loading import BaseSceneLoader
from dronalize.loading.loader import IngestOutput, Source
from dronalize.maps import no_map
from dronalize.maps.resolver import MapResolver, shared_map
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable


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
        self._data_root: Path = self._normalize_data_root(data_root)
        self._batch_size: int | None = file_batch_size
        self._train_dir: Path = self._data_root / "forecasting_train_v1.1" / "train" / "data"
        self._val_dir: Path = self._data_root / "forecasting_val_v1.1" / "val" / "data"
        self._test_dir: Path = self._data_root / "forecasting_test_v1.1" / "test_obs" / "data"

    @override
    def train_sources(self) -> Iterable[Source[list[Path]]]:
        return self._sources_from_dir(self._train_dir)

    @override
    def validate_sources(self) -> Iterable[Source[list[Path]]]:
        return self._sources_from_dir(self._val_dir)

    @override
    def test_sources(self) -> Iterable[Source[list[Path]]]:
        return self._sources_from_dir(self._test_dir)

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
        return sum(self._count_sources_for_split(split) for split in self.selected_splits)

    @override
    def pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .compose(
                trajectory_pipeline(
                    self.loader_config,
                    derivative_rename=self.derivative_names(),
                    forward_fill=["agent_category"],
                )
            )
            .then(tr.yaw_from_vel())
        )

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
        files: list[Path] = sorted(data_dir.glob("*.csv"))
        batch_size: int = self._batch_size or len(files) or 1
        for start in range(0, len(files), batch_size):
            batch_files = files[start : start + batch_size]
            yield Source(identifier=start, inner=batch_files)

    def _count_sources(self, data_dir: Path) -> int:
        num_files = self._count_matching_files([data_dir], "*.csv")
        if num_files == 0:
            return 0
        batch_size = self._batch_size or num_files
        batches, extra = divmod(num_files, batch_size)
        return batches + int(extra > 0)

    def _count_sources_for_split(self, split: DatasetSplit | None) -> int:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self._train_dir)
        if split is DatasetSplit.VAL:
            return self._count_sources(self._val_dir)
        if split is DatasetSplit.TEST:
            return self._count_sources(self._test_dir)
        return (
            self._count_sources(self._train_dir)
            + self._count_sources(self._val_dir)
            + self._count_sources(self._test_dir)
        )


_SCHEMA: pl.Schema = pl.Schema({
    "TIMESTAMP": pl.Float64,
    "TRACK_ID": pl.String,
    "OBJECT_TYPE": pl.Categorical(pl.Categories("AV", "OTHERS")),
    "X": pl.Float64,
    "Y": pl.Float64,
    "CITY_NAME": pl.String,
})
