from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.common.trajectory.basic import yaw_from_vel_expr
from dronalize.common.trajectory.derivative import derivative
from dronalize.common.trajectory.filter import filter_scene_expr
from dronalize.common.trajectory.resample import resample_tracks
from dronalize.core import AgentCategory, BaseSceneLoader, LoaderConfig, Resampling
from dronalize.core.datatypes import map_context as mc
from dronalize.core.protocols.loader import Source

if TYPE_CHECKING:
    from collections.abc import Iterable


class InteractionLoader(BaseSceneLoader[str, list[Path]]):
    """Processor for the INTERACTION dataset."""

    def __init__(
        self,
        data_dir: Path,
        file_batch_size: int | None = None,
        config: LoaderConfig | None = None,
    ) -> None:
        """Initialize the processor.

        The processor will read all CSV files in the given directory, and
        expects them to have the same schema as the INTERACTION dataset.

        Parameters
        ----------
        data_dir : Path
            Directory containing the INTERACTION dataset CSV files.
        file_batch_size : int, optional
            Number of files to read in each batch. If None, all files will be
            read at once. Higher batch size may lead to faster processing at
            diminishing returns, but also higher memory usage. `None` is not
            recommended for large amounts of data.
        config : LoaderConfig, optional
            Processor configuration override. If None, the default
            configuration will be used.

        Raises
        ------
        ValueError
            If `window_params` is set in the config, since
            `InteractionLoader` does not support windowing.

        """
        if config is not None and config.window_params is not None:
            msg = f"InteractionProcessor does not support window_params, but got {config.window_params}"
            raise ValueError(msg)

        super().__init__(loader_config=config, enforce_schema=True)
        self._data_dir = data_dir
        self._file_batch_size: int | None = file_batch_size

    @override
    def sources(self) -> Iterable[Source[str, list[Path]]]:
        csv_files = list(self._data_dir.glob("*.csv"))
        batch_size = self._file_batch_size or len(csv_files)
        for start in range(0, len(csv_files), batch_size):
            batch_files = csv_files[start : start + batch_size]
            yield Source(f"{self._data_dir}_b{start}", batch_files)

    @override
    def num_sources(self) -> int | None:
        if self._file_batch_size is None:
            return 1

        num_files = len(list(self._data_dir.glob("*.csv")))
        return (num_files + self._file_batch_size - 1) // self._file_batch_size

    @override
    def load_raw(
        self, source: Source[str, list[Path]]
    ) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
        resampling = self.loader_config.resampling or Resampling(1, 1)
        data = (
            pl
            .scan_csv(source.inner, include_file_paths="file_id", schema=_SCHEMA)
            .drop("track_to_predict", "interesting_agent", "width", "length", "timestamp_ms")
            .rename({
                "agent_type": "agent_category",
                "psi_rad": "yaw",
                "frame_id": "frame",
                "track_id": "id",
            })
            .with_columns(
                pl.col("file_id").cast(pl.Categorical).to_physical(),
                pl.col("case_id").cast(pl.UInt32),
            )
        )

        data_filtered = data.filter(
            filter_scene_expr(
                self.loader_config,
                group_by=["file_id", "case_id"],
                category_column="agent_category",
            )
        )

        data_filtered = data_filtered.with_columns(
            self._map_agent_category().alias("agent_category"),
        )

        data_processed = resample_tracks(
            data_filtered,
            resampling.up,
            resampling.down,
            group_by=["file_id", "case_id", "id"],
            add_derivative=False,
            add_second_derivative=False,
            method=resampling.method,
            dt=self.loader_config.sample_time,
            derivative_rename=self.derivative_names(),
            forward_fill=["agent_category"],
        )

        for _, group in data_processed.collect().group_by(["file_id", "case_id"]):
            yield group.lazy().drop("file_id", "case_id"), mc.Implicit()

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return derivative(
            df.with_columns(
                pl
                .when(pl.col("yaw").is_null())
                .then(yaw_from_vel_expr("vx", "vy", "yaw"))
                .otherwise(pl.col("yaw")),
            ),
            "vx",
            "vy",
            dt=self.post_sample_time,
            derivative_rename={1: ["ax", "ay"]},
        )

    @override
    def default_config(self) -> LoaderConfig:
        return LoaderConfig(10, 30, 0.1)

    @staticmethod
    def _map_agent_category() -> pl.Expr:
        # "car" -> AgentCategory.CAR
        # "pedestrian/bicycle" -> AgentCategory.PEDESTRIAN if avg speed < 2 m/s else AgentCategory.BICYCLE
        return (
            pl
            .when(pl.col("agent_category") == "car")
            .then(AgentCategory.CAR.value)
            .when(pl.col("agent_category").is_in(["pedestrian/bicycle"]))
            .then(
                pl
                .when((pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt() < 2)
                .then(AgentCategory.PEDESTRIAN.value)
                .otherwise(AgentCategory.BICYCLE.value)
            )
            .otherwise(pl.col("agent_category"))
        )


_SCHEMA = pl.Schema({
    "case_id": pl.Float32,
    "track_id": pl.UInt32,
    "frame_id": pl.UInt32,
    "timestamp_ms": pl.Float32,
    "agent_type": pl.String,
    "x": pl.Float32,
    "y": pl.Float32,
    "vx": pl.Float32,
    "vy": pl.Float32,
    "psi_rad": pl.Float32,
    "length": pl.Float32,
    "width": pl.Float32,
    "track_to_predict": pl.Float32,
    "interesting_agent": pl.Float32,
})


if __name__ == "__main__":
    data_dir = Path("/home/west/Developer/behavior-prediction/datasets/interact/train")

    processor = InteractionLoader(data_dir=data_dir)
    count = 0
    for _scene in processor.scenes():
        if count % 200 == 0:
            print(f"Processed {count} scenes")
        count += 1
    print(f"Total scenes processed: {count}")
