from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.config.loader import LoaderConfig
from dronalize.core.base import BaseSceneLoader
from dronalize.core.categories import AgentCategory
from dronalize.core.interfaces import IngestOutput, Source
from dronalize.core.split import DatasetSplit
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable


class Argoverse2Loader(BaseSceneLoader[list[Path]]):
    """Loader for Argoverse 2 trajectory data stored in Parquet files."""

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
            Path to the root directory of the Argoverse 2 dataset.
            This directory should contain `train/`, `val/`, and `test/`
            subdirectories with Parquet files.
        file_batch_size : int, optional
            Number of files to process in each batch.
        loader_config : LoaderConfig, optional
            Configuration for the loader.
        split : DatasetSplit, optional
            Which dataset split to load. Defaults to all sources.

        """
        super().__init__(loader_config=loader_config, enforce_schema=True, split=split)
        self._data_root = self._normalize_data_root(data_root)
        self._file_batch_size = file_batch_size

    # ------------------------------------------------------------------
    # Split-aware source discovery
    # ------------------------------------------------------------------

    def _sources_from_dir(self, data_dir: Path) -> Iterable[Source[list[Path]]]:
        parquet_files = sorted(data_dir.glob("*/*.parquet"))
        batch_size: int = self._file_batch_size or len(parquet_files)
        for i in range(0, len(parquet_files), batch_size):
            batch_files = parquet_files[i : i + batch_size]
            yield Source(identifier=i, inner=batch_files)

    @override
    def all_sources(self) -> Iterable[Source[list[Path]]]:
        yield from self.train_sources()
        yield from self.validate_sources()
        yield from self.test_sources()

    @override
    def train_sources(self) -> Iterable[Source[list[Path]]]:
        return self._sources_from_dir(self._data_root / "train")

    @override
    def validate_sources(self) -> Iterable[Source[list[Path]]]:
        return self._sources_from_dir(self._data_root / "val")

    @override
    def test_sources(self) -> Iterable[Source[list[Path]]]:
        return self._sources_from_dir(self._data_root / "test")

    # ------------------------------------------------------------------
    # Ingestion / pipeline
    # ------------------------------------------------------------------

    @override
    def ingest(self, source: Source[list[Path]]) -> Iterable[IngestOutput]:
        # Build a mapping from parquet file path to its co-located JSON map path.
        # Each parquet lives in a subdirectory like <scenario_id>/<scenario_id>.parquet
        # alongside a <scenario_id>.json map file.
        file_to_map: dict[str, str] = {}
        for pq in source.inner:
            json_candidates = list(pq.parent.glob("*.json"))
            if json_candidates:
                file_to_map[str(pq)] = str(json_candidates[0])

        batch_lf = pl.scan_parquet(source.inner, include_file_paths="file_id").select(
            pl.col("file_id"),
            self._map_object_type_expr("object_type").alias("agent_category"),
            pl.col("track_id").str.replace("AV", "0").cast(pl.Int32).alias("id"),
            pl.col("timestep").alias("frame"),
            pl.col("position_x").alias("x"),
            pl.col("position_y").alias("y"),
            pl.col("velocity_x").alias("vx"),
            pl.col("velocity_y").alias("vy"),
            pl.col("heading").alias("yaw"),
        )

        for (file_id,), group in batch_lf.collect().group_by(["file_id"]):
            map_path = file_to_map.get(str(file_id))
            yield group.lazy().drop("file_id"), map_path

    @override
    def num_sources(self) -> int | None:
        dirs: list[Path] = []
        split = self._split
        if split in {DatasetSplit.ALL, DatasetSplit.TRAIN}:
            dirs.append(self._data_root / "train")
        if split in {DatasetSplit.ALL, DatasetSplit.VAL}:
            dirs.append(self._data_root / "val")
        if split in {DatasetSplit.ALL, DatasetSplit.TEST}:
            dirs.append(self._data_root / "test")

        num_files = self._count_matching_files(dirs, "*/*.parquet")
        batch_size = self._file_batch_size or num_files or 1
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
        return LoaderConfig(input_len=50, output_len=60, sample_time=0.1).with_filtering(
            require_frames=[49],
            filter_agent_category=[
                AgentCategory.STATIC_OBJECT,
                AgentCategory.UNKNOWN,
                AgentCategory.UNIMPORTANT,
            ],
        )

    @staticmethod
    def _map_object_type_expr(col: str) -> pl.Expr:
        mapping = {
            "static": AgentCategory.STATIC_OBJECT,
            "riderless_bicycle": AgentCategory.STATIC_OBJECT,
            "construction": AgentCategory.STATIC_OBJECT,
            "vehicle": AgentCategory.CAR,
            "motorcyclist": AgentCategory.MOTORCYCLE,
            "cyclist": AgentCategory.BICYCLE,
            "bus": AgentCategory.BUS,
            "pedestrian": AgentCategory.PEDESTRIAN,
            "background": AgentCategory.UNIMPORTANT,
            "unknown": AgentCategory.UNKNOWN,
        }
        return pl.col(col).replace_strict(
            mapping,
            default=AgentCategory.UNKNOWN,
            return_dtype=pl.Int32,
        )
