from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config import LoaderConfig
from dronalize.loading import BaseSceneLoader, IngestOutput, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline
from dronalize.scene import POSITIONS_VELOCITY_YAW_V1

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.config.map import MapConfig
    from dronalize.scene import SceneSchema


class InteractionLoader(BaseSceneLoader[list[Path]]):
    """Loader for the INTERACTION dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        *,
        file_batch_size: int | None = None,
    ) -> None:
        """Initialize the dataset loader.

        The loader reads all CSV files in the given directory and
        expects them to have the same schema as the INTERACTION dataset.

        Parameters
        ----------
        data_root : Path or str
            Path to the root of the INTERACTION dataset.  This directory
            should contain `train/`, `val/`, `test_multi-agent/`,
            and `test_conditional-multi-agent/` subdirectories.
        loader_config : , optional
            Loader configuration override. If None, the default configuration
            is used.
        map_config : MapConfig, optional
            Map configuration override. If None, the default configuration is
            used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. Can contain one or more predefined splits,
            or `None` to process all sources.
        file_batch_size : int, optional
            Number of files to read in each batch. If None, all files will be
            read at once. Higher batch size may lead to faster processing at
            diminishing returns, but also higher memory usage. `None` is not
            recommended for large amounts of data.

        """
        super().__init__(loader_config=loader_config, map_config=map_config, splits=splits)
        self._data_root: Path = Path(data_root)
        self._file_batch_size: int | None = file_batch_size

    @classmethod
    @override
    def predefined_splits(cls) -> tuple[DatasetSplit, ...]:
        return (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

    def _sources_from_dir(self, data_dir: Path) -> Iterable[Source[list[Path]]]:
        if not data_dir.is_dir():
            return
        csv_files = sorted(data_dir.rglob("*.csv"))
        batch_size = self._file_batch_size or len(csv_files) or 1
        for start in range(0, len(csv_files), batch_size):
            batch_files = csv_files[start : start + batch_size]
            yield Source(f"{data_dir.name}_b{start}", batch_files)

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[list[Path]]]:
        if split is DatasetSplit.TRAIN:
            yield from self._sources_from_dir(self._data_root / "train")
            return
        if split is DatasetSplit.VAL:
            yield from self._sources_from_dir(self._data_root / "val")
            return
        if split is DatasetSplit.TEST:
            yield from self._sources_from_dir(self._data_root / "test_multi-agent")
            yield from self._sources_from_dir(self._data_root / "test_conditional-multi-agent")
            return

    @override
    def ingest(self, source: Source[list[Path]]) -> Iterable[IngestOutput]:
        data = (
            pl
            .scan_csv(source.inner, include_file_paths="file_id", schema=_SCHEMA)
            .drop("track_to_predict", "interesting_agent", "width", "length", "timestamp_ms")
            .rename({
                "psi_rad": "yaw",
                "frame_id": "frame",
                "track_id": "id",
            })
            .with_columns(
                pl.col("file_id").cast(pl.Categorical).to_physical(),
                pl.col("case_id").cast(pl.UInt32),
                self._map_agent_category().alias("agent_category"),
            )
            .drop("agent_type")
        )

        for _, group in data.collect().group_by(["file_id", "case_id"]):
            yield group.lazy().drop("file_id", "case_id"), None

    @override
    def num_sources(self) -> int | None:
        splits = self.splits if self.splits is not None else self.predefined_splits()
        return sum(self._count_sources_for_split(split) for split in splits)

    @override
    def pipeline(self) -> Pipeline:
        return Pipeline().compose(trajectory_pipeline(self.loader_config))

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_VELOCITY_YAW_V1

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(input_len=10, output_len=30, sample_time=0.1).with_filtering(
            require_frames=[19]
        )

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
                .otherwise(AgentCategory.BICYCLE.value),
            )
            .otherwise(pl.col("agent_type"))
        )

    def _count_sources(self, data_dir: Path) -> int:
        if not data_dir.is_dir():
            return 0
        num_files = sum(1 for _ in data_dir.rglob("*.csv"))
        if num_files == 0:
            return 0
        batch_size = self._file_batch_size or num_files
        batches, extra = divmod(num_files, batch_size)
        return batches + int(extra > 0)

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self._data_root / "train")
        if split is DatasetSplit.VAL:
            return self._count_sources(self._data_root / "val")
        return self._count_sources(self._data_root / "test_multi-agent") + self._count_sources(
            self._data_root / "test_conditional-multi-agent"
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
