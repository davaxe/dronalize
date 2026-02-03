# Copyright 2024-2025, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

import polars as pl

from preprocessing.trajectory.interface import (
    Category,
    DataProcessor,
    ProcessorConfig,
    Resampling,
)
from preprocessing.trajectory.resample import resample_tracks
from preprocessing.trajectory.utils import (
    filter_scene_expr,
    sliding_window,
    yaw_from_vel,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class EthUcyProcessor(DataProcessor[str, pl.LazyFrame]):
    """Processor for ETH/UCY pedestrian trajectory datasets."""

    def __init__(
        self,
        data_root: Path,
        dataset: str | Sequence[str],
        config: ProcessorConfig | None = None,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        """Initialize with the given configuration."""
        if config is None:
            config = self._default_config()

        super().__init__(processor_config=config, validate_output=True)
        self._data_root = data_root
        self._dataset = {dataset} if isinstance(dataset, str) else set(dataset)
        self._split = split
        self._window_params = self.processor_config.window_params
        self._filtering_params = self.processor_config.scene_filtering

    @override
    def sources(self) -> Iterable[tuple[str, pl.LazyFrame]]:
        for dataset in self._dataset:
            data_dir = self._data_root / dataset / self._split
            # Sort to ensure consistent order across runs and systems.
            for data_file in sorted(data_dir.iterdir()):
                yield (data_file.name, self._read_data_file(data_file))

    @override
    def load_raw(self, source: pl.LazyFrame) -> Iterable[pl.LazyFrame]:
        # Remove single-frame pedestrians to prevent resampling errors
        source = source.filter(pl.col("frame").n_unique().over("id") > 1)
        resampling = self.processor_config.resampling or Resampling(1, 1)
        group_by: list[str] = []
        # Apply sliding window logic if parameters are present
        if self._window_params is not None:
            source = sliding_window(
                source,
                window_size=self.sequence_length,
                step_size=self._window_params.step_size,
                sliding_col="frame",
                is_sorted=True,
                return_iterable=False,
            )
            group_by.append("window_index")

        source_filtered = source.filter(
            filter_scene_expr(
                self.processor_config,
                group_by=group_by[-1] if len(group_by) > 0 else None,
            )
        )
        group_by.append("id")

        processed_source = resample_tracks(
            source_filtered,
            resampling.up,
            resampling.down,
            group_by=group_by,
            add_derivative=True,
            add_second_derivative=True,
            method=resampling.method,
            dt=self.processor_config.sample_time,
            derivative_rename=self.derivative_names(),
        )
        if self._window_params is None:
            yield processed_source
            return

        for _, group in processed_source.collect().group_by("window_index"):
            yield group.lazy()

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return yaw_from_vel(df, yaw_col="yaw").with_columns(
            pl.lit(Category.PEDESTRIAN).alias("agent_class"),
        )

    def _read_data_file(self, path: Path) -> pl.LazyFrame:
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

    @staticmethod
    def _default_config() -> ProcessorConfig:
        return (
            ProcessorConfig(
                input_len=8,
                output_len=12,
                sample_time=0.4,
            )
            .window_parameters(step_size=1)
            .scene_filtering_parameters(require_all_valid=True)
            .resampling_parameters(4, 1, method="spline")
        )


if __name__ == "__main__":
    processor = EthUcyProcessor(
        data_root=Path("data"), dataset="hotel", split="test"
    )
    start_time = time.perf_counter()
    count: int = 0
    total_time = 0.0
    for _scene in processor.scenes_iter():
        count += 1

    end_time = time.perf_counter()
    print(f"Processed {count} scenes in {end_time - start_time:.2f} seconds.")
