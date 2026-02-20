from collections.abc import Iterable
from pathlib import Path

import polars as pl
from typing_extensions import override

from preprocessing.common.trajectory_utils.basic import yaw_from_vel
from preprocessing.common.trajectory_utils.filter import filter_scene_expr
from preprocessing.common.trajectory_utils.resample import resample_tracks
from preprocessing.core.categories import AgentCategory
from preprocessing.core.interface import DataProcessor, ProcessorConfig, Resampling


class Argoverse1Processor(DataProcessor[str, pl.LazyFrame]):
    """Processor for Argoverse 1 dataset stored in CSV format."""

    def __init__(
        self,
        data_path: Path,
        file_batch_size: int | None = 100,
        config: ProcessorConfig | None = None,
    ) -> None:
        """Initialize the data processor.

        Args:
            data_path: path to the directory of CSV files.
            file_batch_size: number of files to read in each batch. If None, all files will be read
                at once. Higher batch size may lead to faster processing at diminishing returns, but
                also higher memory usage. `None` is not recommended for large amount of data.
            config: processor configuration override. If None, the default configuration will be used.

        """
        super().__init__(processor_config=config, enforce_schema=True)
        self._data_path = data_path
        self._batch_size: int | None = file_batch_size

    @override
    def sources(self) -> Iterable[tuple[int, pl.LazyFrame]]:
        files: list[Path] = sorted(self._data_path.glob("*.csv"))
        batch_size: int = self._batch_size or len(files)
        for start in range(0, len(files), batch_size):
            batch_files = files[start : start + batch_size]
            extra: list[pl.Expr] = []
            if len(batch_files) > 1:
                extra.append(pl.col("file_id").cast(pl.Categorical).to_physical())

            yield (
                start,
                pl
                .scan_csv(batch_files, include_file_paths="file_id", schema=_SCHEMA)
                .with_columns(
                    pl
                    .when(pl.col("OBJECT_TYPE") == "AV")
                    .then(AgentCategory.CAR)
                    .otherwise(AgentCategory.UNKNOWN)
                    .alias("agent_category"),
                    pl.col("TRACK_ID").rank(method="dense").sub(1).cast(pl.Int64).alias("id"),
                    pl.col("TIMESTAMP").rank(method="dense").sub(1).cast(pl.Int64).alias("frame"),
                    *extra,
                )
                .drop("OBJECT_TYPE", "TRACK_ID", "TIMESTAMP")
                .rename({"X": "x", "Y": "y", "CITY_NAME": "map"}),
            )

    @override
    def load_raw(self, source: pl.LazyFrame) -> Iterable[pl.LazyFrame]:
        resampling = self.processor_config.resampling or Resampling(1, 1)

        source_filtered = source.filter(
            filter_scene_expr(
                self.processor_config,
                group_by=["file_id"],
                category_column="agent_category",
            )
        )

        source_resampled = resample_tracks(
            source_filtered,
            resampling.up,
            resampling.down,
            group_by=["file_id"],
            add_derivative=True,
            add_second_derivative=True,
            method=resampling.method,
            dt=self.processor_config.sample_time,
            derivative_rename=self.derivative_names(),
            forward_fill=["agent_category"],
        )
        for _, group in source_resampled.collect().group_by(["file_id"]):
            yield yaw_from_vel(group.lazy()).drop("file_id")

    @override
    def normalize(self, scene: pl.LazyFrame) -> pl.LazyFrame:
        self.attach_to_scene(select_expr={"map_information": pl.col("map").first()}, properties={})
        return yaw_from_vel(scene, yaw_col="yaw")

    @override
    def default_config(self) -> ProcessorConfig:
        return ProcessorConfig(20, 30, 0.1)


_SCHEMA: pl.Schema = pl.Schema({
    "TIMESTAMP": pl.Float64,
    "TRACK_ID": pl.String,
    "OBJECT_TYPE": pl.Categorical(pl.Categories("AV", "OTHERS")),
    "X": pl.Float32,
    "Y": pl.Float32,
    "CITY_NAME": pl.String,
})

if __name__ == "__main__":
    import time

    data_path = Path(
        "/home/west/Developer/behavior-prediction/datasets/argoverse/forecasting_train_v1.1/train/data"
    )
    processor = Argoverse1Processor(data_path, file_batch_size=1000)
    count = 0
    time_start = time.perf_counter()
    for scene in processor.scenes_iter():
        count += 1
        if count % 500 == 0:
            print(f"Processed {count} scenes in {time.perf_counter() - time_start:.2f} seconds")

    print(f"Total scenes processed: {count}")
    print(f"Time taken: {time.perf_counter() - time_start:.2f} seconds")
