from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.core.transforms as tr
from dronalize.core import AgentCategory, BaseSceneLoader, LoaderConfig
from dronalize.core.pipeline import Pipeline
from dronalize.core.pipelines import trajectory_pipeline
from dronalize.core.protocols.loader import IngestOutput, Source

if TYPE_CHECKING:
    from collections.abc import Iterable


class InteractionLoader(BaseSceneLoader[str, list[Path]]):
    """Processor for the INTERACTION dataset."""

    def __init__(
        self,
        data_dir: Path,
        file_batch_size: int | None = None,
        loader_config: LoaderConfig | None = None,
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
        loader_config : LoaderConfig, optional
            Processor configuration override. If None, the default
            configuration will be used.

        Raises
        ------
        ValueError
            If `window_params` is set in the config, since
            `InteractionLoader` does not support windowing.

        """
        if loader_config is not None and loader_config.window_params is not None:
            msg = (
                f"InteractionProcessor does not support window_params, "
                f"but got {loader_config.window_params}"
            )
            raise ValueError(msg)

        super().__init__(loader_config=loader_config, enforce_schema=True)
        self._data_dir = data_dir
        self._file_batch_size: int | None = file_batch_size

    @override
    def sources(self) -> Iterable[Source[str, list[Path]]]:
        csv_files = sorted(self._data_dir.glob("*.csv"))
        batch_size = self._file_batch_size or len(csv_files)
        for start in range(0, len(csv_files), batch_size):
            batch_files = csv_files[start : start + batch_size]
            yield Source(f"{self._data_dir}_b{start}", batch_files)

    @override
    def ingest(self, source: Source[str, list[Path]]) -> Iterable[IngestOutput]:
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
        if self._file_batch_size is None:
            return 1

        num_files = len(list(self._data_dir.glob("*.csv")))
        return (num_files + self._file_batch_size - 1) // self._file_batch_size

    @override
    def pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .compose(
                trajectory_pipeline(
                    self.loader_config,
                    derivative_rename=self.derivative_names(),
                    add_derivative=False,
                    add_second_derivative=False,
                    forward_fill=["agent_category"],
                )
            )
            .then(tr.yaw_from_vel(only_null=True))
            .then(
                tr.derivative(
                    "vx",
                    "vy",
                    dt=self.post_sample_time,
                    derivative_rename={1: ["ax", "ay"]},
                )
            )
        )

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(10, 30, 0.1).with_filtering(require_frames=[19])

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
