from collections.abc import Iterable
from pathlib import Path

import polars as pl
from typing_extensions import override

from dronalize.common.trajectory.basic import yaw_from_vel
from dronalize.common.trajectory.filter import filter_scene_expr
from dronalize.common.trajectory.resample import Resampling, resample_tracks
from dronalize.core.datatypes import map_context as mc
from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.protocols.loader import BaseSceneLoader, LoaderConfig, Source


class Argoverse1Loader(BaseSceneLoader[int, pl.LazyFrame]):
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
    def sources(self) -> Iterable[Source[int, pl.LazyFrame]]:
        files: list[Path] = sorted(self._data_path.glob("*.csv"))
        batch_size: int = self._batch_size or len(files)
        for start in range(0, len(files), batch_size):
            batch_files = files[start : start + batch_size]
            yield Source(
                identifier=start,
                inner=pl
                .scan_csv(batch_files, include_file_paths="file_id", schema=_SCHEMA)
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
                .rename({"X": "x", "Y": "y", "CITY_NAME": "map"}),
            )

    @override
    def num_sources(self) -> int | None:
        num_files = sum(1 for _ in self._data_path.glob("*.csv"))
        batch_size = self._batch_size or num_files
        batches, extra = divmod(num_files, batch_size)
        return batches + int(extra > 0)

    @override
    def load_raw(
        self, source: Source[int, pl.LazyFrame]
    ) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
        resampling = self.loader_config.resampling or Resampling(1, 1)

        source_filtered = source.inner.filter(
            filter_scene_expr(
                *self.loader_config.filter_args(),
                group_by=["file_id"],
                category_column="agent_category",
            )
        )

        source_resampled = resample_tracks(
            source_filtered,
            resampling,
            group_by=["file_id"],
            add_derivative=True,
            add_second_derivative=True,
            dt=self.loader_config.sample_time,
            derivative_rename=self.derivative_names(),
            forward_fill=["agent_category"],
        )
        for _, group in source_resampled.collect().group_by(["file_id"]):
            yield (
                yaw_from_vel(group.lazy()).drop("file_id"),
                mc.Explicit(map=str(group["map"].first())),
            )

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return yaw_from_vel(df, yaw_col="yaw")

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(20, 30, 0.1)


_SCHEMA: pl.Schema = pl.Schema({
    "TIMESTAMP": pl.Float64,
    "TRACK_ID": pl.String,
    "OBJECT_TYPE": pl.Categorical(pl.Categories("AV", "OTHERS")),
    "X": pl.Float32,
    "Y": pl.Float32,
    "CITY_NAME": pl.String,
})
