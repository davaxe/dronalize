from collections.abc import Iterable
from pathlib import Path

import polars as pl
from typing_extensions import override

from dronalize.common.trajectory.basic import yaw_from_vel
from dronalize.common.trajectory.derivative import derivative
from dronalize.common.trajectory.filter import filter_scene_expr
from dronalize.common.trajectory.resample import Resampling, resample_tracks
from dronalize.core.datatypes import map_context as mc
from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.protocols.loader import BaseSceneLoader, LoaderConfig, Source

# TODO: Currently the column "focal_agent_id" is disgarded and not used; might want to provide a way
# to identify it downstream. Either implcitlty by assigning a specific id or explicitly by providing
# a way to specify it.


class Argoverse2Loader(BaseSceneLoader[int, pl.LazyFrame]):
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
        self._unique_types: set[str] = set()

    @override
    def sources(self) -> Iterable[Source[int, pl.LazyFrame]]:
        # Sort to match files correctly
        parquet_files = sorted(str(f) for f in self._data_dir.glob("*/*.parquet"))
        json_files = sorted(str(f) for f in self._data_dir.glob("*/*.json"))
        map_lookup = pl.DataFrame({"file_id": parquet_files, "map_path": json_files}).lazy()

        batch_size: int = self._file_batch_size or len(parquet_files)
        for i in range(0, len(parquet_files), batch_size):
            batch_files: list[str] = list(parquet_files[i : i + batch_size])
            yield Source(
                identifier=i,
                inner=pl
                .scan_parquet(batch_files, include_file_paths="file_id")
                .join(map_lookup, on="file_id", how="left")
                .with_columns(
                    pl.col("file_id").cast(pl.Categorical).to_physical(),
                    self._map_object_type_expr("object_type").alias("agent_category"),
                    pl.col("track_id").str.replace("AV", "0").cast(pl.Int32),
                )
                .select(
                    pl.col("file_id").cast(pl.Categorical).to_physical(),
                    self._map_object_type_expr("object_type").alias("agent_category"),
                    pl.col("track_id").str.replace("AV", "0").cast(pl.Int32).alias("id"),
                    pl.col("timestep").alias("frame"),
                    pl.col("position_x").alias("x"),
                    pl.col("position_y").alias("y"),
                    pl.col("velocity_x").alias("vx"),
                    pl.col("velocity_y").alias("vy"),
                    pl.col("heading").alias("yaw"),
                ),
            )

    @override
    def num_sources(self) -> int | None:
        num_files = sum(1 for _ in self._data_dir.glob("*/*.parquet"))
        batch_size = self._file_batch_size or num_files
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

        if resampling.no_resampling:
            source_resampled = derivative(
                source_filtered,
                "vx",
                "vy",
                dt=self.loader_config.sample_time,
                group_by=["file_id"],
                derivative_rename={1: ["ax", "ay"]},
            )
        else:
            source_resampled = yaw_from_vel(
                resample_tracks(
                    source_filtered,
                    resampling,
                    group_by=["file_id"],
                    add_derivative=True,
                    add_second_derivative=True,
                    dt=self.loader_config.sample_time,
                    derivative_rename=self.derivative_names(),
                    forward_fill=["agent_category"],
                )
            )

        for _, group in source_resampled.collect().group_by(["file_id"]):
            yield group.lazy(), mc.Explicit(map_path=str(group["map_path"].first()))

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(50, 60, 0.1).with_filtering(
            filter_agent_category=[
                AgentCategory.STATIC_OBJECT,
                AgentCategory.UNKNOWN,
                AgentCategory.UNIMPORTANT,
            ]
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
            mapping, default=AgentCategory.UNKNOWN, return_dtype=pl.Int32
        )
