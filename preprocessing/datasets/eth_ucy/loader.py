from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from typing_extensions import override

from preprocessing.common.trajectory.basic import yaw_from_vel
from preprocessing.common.trajectory.process import prepare_agent_trajectories
from preprocessing.core import AgentCategory, BaseSceneLoader, LoaderConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from preprocessing.core.datatypes import map_context as mc


class EthUcyLoader(BaseSceneLoader[str, Path]):
    """Processor for ETH/UCY pedestrian trajectory datasets."""

    def __init__(
        self,
        data_root: Path,
        dataset: str | Sequence[str],
        config: LoaderConfig | None = None,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        """Initialize with the given configuration."""
        super().__init__(loader_config=config, enforce_schema=True)
        self._data_root = data_root
        self._dataset = {dataset} if isinstance(dataset, str) else set(dataset)
        self._split = split
        self._window_params = self.loader_config.window_params
        self._filtering_params = self.loader_config.scene_filtering

    @override
    def sources(self) -> Iterable[tuple[str, Path]]:
        for dataset in self._dataset:
            data_dir = self._data_root / dataset / self._split
            # Sort to ensure consistent order across runs and systems.
            for data_file in sorted(data_dir.iterdir()):
                yield data_file.name, data_file

    @override
    def num_sources(self) -> int | None:
        num_sources: int = 0
        for dataset in self._dataset:
            data_dir = self._data_root / dataset / self._split
            num_sources += sum(1 for _ in data_dir.iterdir())

        return num_sources

    @override
    def load_raw(self, source: Path) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
        source_data = EthUcyLoader._read_data_file(source)
        for df in prepare_agent_trajectories(
            source_data,
            self.loader_config,
            add_derivative=True,
            add_second_derivative=True,
            sliding_col="frame",
            agent_category_col=None,
            derivative_rename=self.derivative_names(),
        ):
            yield df, None

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return yaw_from_vel(df, yaw_col="yaw").with_columns(
            pl.lit(AgentCategory.PEDESTRIAN).alias("agent_category"),
        )

    @override
    def default_config(self) -> LoaderConfig:
        return (
            LoaderConfig(
                input_len=8,
                output_len=12,
                sample_time=0.4,
            )
            .window_parameters(step_size=1)
            .scene_filtering_parameters(require_all_valid=True)
            .resampling_parameters(4, 1, method="spline")
        )

    @staticmethod
    def _read_data_file(path: Path) -> pl.LazyFrame:
        return pl.scan_csv(
            path,
            has_header=False,
            separator="\t",
            new_columns=["frame", "id", "x", "y"],
            schema={
                "frame": pl.Int32,
                "id": pl.Int32,
                "x": pl.Float64,
                "y": pl.Float64,
            },
        ).with_columns(
            ((pl.col("frame") - pl.col("frame").min()) // 10).cast(pl.Int32),
            pl.col("id").cast(pl.Int32),
        )


if __name__ == "__main__":
    processor = EthUcyLoader(data_root=Path("data"), dataset="hotel", split="train")
    start_time = time.perf_counter()
    count: int = 0
    total_time = 0.0
    for _scene in processor.scenes():
        count += 1

    end_time = time.perf_counter()
    print(f"Processed {count} scenes in {end_time - start_time:.2f} seconds.")
