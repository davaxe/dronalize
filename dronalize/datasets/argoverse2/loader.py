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


class Argoverse2Loader(BaseSceneLoader[int, list[Path]]):
    """Processor for Argoverse2 trajectory data stored in Parquet files."""

    def __init__(
        self,
        data_dir: Path,
        file_batch_size: int | None = 100,
        loader_config: LoaderConfig | None = None,
    ) -> None:
        """Initialize the Argoverse2TrajectoryProcessor.

        Parameters
        ----------
        data_dir : Path
            Path to the directory containing the Parquet files.
        file_batch_size : int, optional
            Number of files to process in each batch.
        loader_config : LoaderConfig, optional
            Configuration for the loader.

        """
        super().__init__(loader_config=loader_config, enforce_schema=True)
        self._data_dir = data_dir
        self._file_batch_size = file_batch_size

    @override
    def sources(self) -> Iterable[Source[int, list[Path]]]:
        parquet_files = sorted(self._data_dir.glob("*/*.parquet"))
        batch_size: int = self._file_batch_size or len(parquet_files)
        for i in range(0, len(parquet_files), batch_size):
            batch_files = parquet_files[i : i + batch_size]
            yield Source(identifier=i, inner=batch_files)

    @override
    def ingest(self, source: Source[int, list[Path]]) -> Iterable[IngestOutput]:
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
        num_files = sum(1 for _ in self._data_dir.glob("*/*.parquet"))
        batch_size = self._file_batch_size or num_files
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
        return LoaderConfig(50, 60, 0.1).with_filtering(
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
