from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.datatypes.loader_config import LoaderConfig
from dronalize.core.datatypes.split import DatasetSplit
from dronalize.core.protocols.loader import BaseSceneLoader, IngestOutput, Source
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
        *,
        file_batch_size: int | None = 100,
        split: DatasetSplit | None = None,
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
        split : DatasetSplit, optional
            Which dataset split to load. Defaults to all sources.

        """
        super().__init__(loader_config=loader_config, enforce_schema=True, split=split)
        self._data_root = self._normalize_data_root(data_root)
        self._batch_size: int | None = file_batch_size

    @override
    def all_sources(self) -> Iterable[Source[list[Path]]]:
        yield from self.train_sources()
        yield from self.validate_sources()
        yield from self.test_sources()

    @override
    def train_sources(self) -> Iterable[Source[list[Path]]]:
        return self._sources_from_dir(self._data_root / "forecasting_train_v1.1" / "train" / "data")

    @override
    def validate_sources(self) -> Iterable[Source[list[Path]]]:
        return self._sources_from_dir(self._data_root / "forecasting_val_v1.1" / "val" / "data")

    @override
    def test_sources(self) -> Iterable[Source[list[Path]]]:
        return self._sources_from_dir(
            self._data_root / "forecasting_test_v1.1" / "test_obs" / "data"
        )

    # ------------------------------------------------------------------
    # Ingestion / pipeline
    # ------------------------------------------------------------------

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
        dirs: list[Path] = []
        split = self._split
        if split in {DatasetSplit.ALL, DatasetSplit.TRAIN}:
            dirs.append(self._data_root / "forecasting_train_v1.1" / "train" / "data")
        if split in {DatasetSplit.ALL, DatasetSplit.VAL}:
            dirs.append(self._data_root / "forecasting_val_v1.1" / "val" / "data")
        if split in {DatasetSplit.ALL, DatasetSplit.TEST}:
            dirs.append(self._data_root / "forecasting_test_v1.1" / "test_obs" / "data")

        num_files = self._count_matching_files(dirs, "*.csv")
        batch_size = self._batch_size or num_files or 1
        batches, extra = divmod(num_files, batch_size)
        return batches + int(extra > 0)

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

    def _sources_from_dir(self, data_dir: Path) -> Iterable[Source[list[Path]]]:
        files: list[Path] = sorted(data_dir.glob("*.csv"))
        batch_size: int = self._batch_size or len(files)
        for start in range(0, len(files), batch_size):
            batch_files = files[start : start + batch_size]
            yield Source(identifier=start, inner=batch_files)


_SCHEMA: pl.Schema = pl.Schema({
    "TIMESTAMP": pl.Float64,
    "TRACK_ID": pl.String,
    "OBJECT_TYPE": pl.Categorical(pl.Categories("AV", "OTHERS")),
    "X": pl.Float32,
    "Y": pl.Float32,
    "CITY_NAME": pl.String,
})
