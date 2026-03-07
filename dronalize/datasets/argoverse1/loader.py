from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.core.transforms as tr
from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.pipeline import Pipeline
from dronalize.core.pipelines import trajectory_pipeline
from dronalize.core.protocols.loader import BaseSceneLoader, IngestOutput, LoaderConfig, Source

if TYPE_CHECKING:
    from collections.abc import Iterable


class Argoverse1Loader(BaseSceneLoader[int, list[Path]]):
    """Processor for Argoverse 1 dataset stored in CSV format."""

    def __init__(
        self,
        data_dir: Path,
        file_batch_size: int | None = 100,
        loader_config: LoaderConfig | None = None,
    ) -> None:
        """Initialize the data processor.

        Parameters
        ----------
        data_dir : Path
            Path to the directory of CSV files.
        file_batch_size : int, optional
            Number of files to read in each batch. If None, all files will be
            read at once. Higher batch size may lead to faster processing at
            diminishing returns, but also higher memory usage. `None` is not
            recommended for large amounts of data.
        loader_config : LoaderConfig, optional
            Processor configuration override. If None, the default
            configuration will be used.

        """
        super().__init__(loader_config=loader_config, enforce_schema=True)
        self._data_path = data_dir
        self._batch_size: int | None = file_batch_size

    @override
    def sources(self) -> Iterable[Source[int, list[Path]]]:
        files: list[Path] = sorted(self._data_path.glob("*.csv"))
        batch_size: int = self._batch_size or len(files)
        for start in range(0, len(files), batch_size):
            batch_files = files[start : start + batch_size]
            yield Source(identifier=start, inner=batch_files)

    @override
    def ingest(self, source: Source[int, list[Path]]) -> Iterable[IngestOutput]:
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
        num_files = sum(1 for _ in self._data_path.glob("*.csv"))
        batch_size = self._batch_size or num_files
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
        return LoaderConfig(20, 30, 0.1).with_filtering(require_frames=[19])


_SCHEMA: pl.Schema = pl.Schema({
    "TIMESTAMP": pl.Float64,
    "TRACK_ID": pl.String,
    "OBJECT_TYPE": pl.Categorical(pl.Categories("AV", "OTHERS")),
    "X": pl.Float32,
    "Y": pl.Float32,
    "CITY_NAME": pl.String,
})
