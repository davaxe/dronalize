from collections.abc import Iterable
from pathlib import Path

import polars as pl
from typing_extensions import override

from preprocessing.common.trajectory_utils.basic import yaw_from_vel
from preprocessing.common.trajectory_utils.derivative import derivative
from preprocessing.common.trajectory_utils.filter import filter_scene_expr
from preprocessing.common.trajectory_utils.resample import resample_tracks
from preprocessing.core import map_context as mc
from preprocessing.core.categories import AgentCategory
from preprocessing.core.interface import BaseSceneLoader, LoaderConfig, Resampling

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

        Args:
            data_dir: Path to the directory containing the Parquet files.
            file_batch_size: Number of files to process in each batch.
            loader_config: Configuration for the loader.

        """
        super().__init__(loader_config=loader_config, enforce_schema=True)
        self._data_dir = data_dir
        self._file_batch_size = file_batch_size
        self._unique_types: set[str] = set()

    @override
    def sources(self) -> Iterable[tuple[int, pl.LazyFrame]]:
        # Sort to match files correctly
        parquet_files = sorted(str(f) for f in self._data_dir.glob("*/*.parquet"))
        json_files = sorted(str(f) for f in self._data_dir.glob("*/*.json"))
        map_lookup = pl.DataFrame({"file_id": parquet_files, "map_path": json_files}).lazy()

        batch_size: int = self._file_batch_size or len(parquet_files)
        for i in range(0, len(parquet_files), batch_size):
            batch_files: list[str] = list(parquet_files[i : i + batch_size])
            yield (
                i,
                pl
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
    def load_raw(self, source: pl.LazyFrame) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
        resampling = self.loader_config.resampling or Resampling(1, 1)
        source_filtered = source.filter(
            filter_scene_expr(
                self.loader_config,
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
                    resampling.up,
                    resampling.down,
                    group_by=["file_id"],
                    add_derivative=True,
                    add_second_derivative=True,
                    method=resampling.method,
                    dt=self.loader_config.sample_time,
                    derivative_rename=self.derivative_names(),
                    forward_fill=["agent_category"],
                )
            )

        for _, group in source_resampled.collect().group_by(["file_id"]):
            yield group.lazy(), mc.Explicit(str(group["map_path"].first()))

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df

    @override
    def default_config(self) -> LoaderConfig:
        return LoaderConfig(50, 60, 0.1).scene_filtering_parameters(
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


if __name__ == "__main__":
    data_dir = Path("/home/west/Developer/behavior-prediction/datasets/av2/train")
    file_batch_size = 100
    processor = Argoverse2Loader(data_dir, file_batch_size)
    count: int = 0
    import time

    start_time = time.perf_counter()
    for scene in processor.scenes():
        count += 1
        if count % 200 == 0:
            print(scene.map_context)
            print(f"Processed {count} scenes in {time.perf_counter() - start_time:.2f} seconds")

    print(f"Processed {count} scenes in {time.perf_counter() - start_time:.2f} seconds")
