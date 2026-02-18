from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from preprocessing.common.trajectory_utils import derivative, filter_scene_expr, resample_tracks
from preprocessing.common.trajectory_utils.basic import yaw_from_vel_expr
from preprocessing.core import AgentCategory
from preprocessing.core.interface import DataProcessor, ProcessorConfig, Resampling

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass
class _Source:
    csv_files: list[Path]


class InteractionProcessor(DataProcessor[str, _Source]):
    """Processor for the INTERACTION dataset."""

    def __init__(self, data_dir: Path, config: ProcessorConfig | None = None) -> None:
        """Initialize the processor.

        The processor will read all CSV files in the given directory, and expects them to have the
        same schema as the INTERACTION dataset.

        Args:
            data_dir: directory containing the INTERACTION dataset CSV files.
            config: processor configuration override. If None, the default configuration will be used.

        Raises:
            ValueError: if window_params is set in the config, since InteractionProcessor does not
                support windowing.

        """
        if config is not None and config.window_params is not None:
            msg = f"InteractionProcessor does not support window_params, but got {config.window_params}"
            raise ValueError(msg)

        super().__init__(config, enforce_schema=True)
        self._data_dir = data_dir

    @override
    def sources(self) -> Iterable[tuple[str, _Source]]:
        csv_files = list(self._data_dir.glob("*.csv"))
        yield str(self._data_dir), _Source(csv_files=csv_files)

    @override
    def load_raw(self, source: _Source) -> Iterable[pl.LazyFrame]:
        resampling = self.processor_config.resampling or Resampling(1, 1)
        data = (
            pl
            .scan_csv(
                *[source.csv_files],
                include_file_paths="file_id",
                schema=_SCHEMA,
            )
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
                self.processor_config,
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
            group_by=["file_id", "case_id"],
            add_derivative=False,
            add_second_derivative=False,
            method=resampling.method,
            dt=self.processor_config.sample_time,
            derivative_rename=self.derivative_names(),
            forward_fill=["agent_category"],
        )

        for _, group in data_processed.collect().group_by(["file_id", "case_id"]):
            yield group.lazy().drop("file_id", "case_id")

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
    def default_config(self) -> ProcessorConfig:
        return ProcessorConfig(10, 30, 0.1)

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

    processor = InteractionProcessor(data_dir=data_dir)
    count = 0
    for scene in processor.scenes_iter():
        if count % 200 == 0:
            print(f"Processed {count} scenes")
        count += 1
    print(f"Total scenes processed: {count}")
