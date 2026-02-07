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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

import polars as pl

from preprocessing.trajectory.interface import Category, DataProcessor, Frame
from preprocessing.trajectory.resample import resample_tracks
from preprocessing.trajectory.utils import yaw_from_vel

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for pedestrian data loading."""

    data_root: Path
    """Root directory for the pedestrian data, should contain subdirectories for
    each dataset."""

    dataset: str | set[str]
    """Name of the dataset to load, e.g., 'eth', 'hotel', 'univ', 'zara1'."""

    split: Literal["train", "val", "test"] = "train"
    """Data split to load, can be 'train', 'val', or 'test'."""

    org_obs_len: int = 8
    """Length of the observation sequence in frames."""

    org_pred_len: int = 12
    """Length of the prediction sequence in frames."""

    min_pedestrian: int = 2
    """Minimum number of pedestrians required in a sequence to be valid."""

    org_sample_time: float = 0.4
    """Time interval between frames in seconds."""

    require_all_valid: bool = True
    """If True, requires all pedestrians in a sequence to have valid positions."""

    target_sample_time: float = 0.1
    """The target time interval for resampling the trajectories, in seconds."""

    multiple_targets_per_window: bool = False
    """If True, allows multiple target pedestrians per sequence window."""

    window_skip: int = 1
    """Number of samples between sliding windows in the sequence."""

    sep: str = "\t"
    """Separator used in the csv-like data files"""


class EthUcyProcessor(DataProcessor[str, pl.DataFrame, Frame]):
    """Processor for ETH/UCY pedestrian trajectory datasets."""

    def __init__(self, config: Config) -> None:
        """Initialize with the given configuration."""
        super().__init__(validate_output=True)
        self.config = config

    @property
    def sequence_length(self) -> int:
        """Total sequence length (observation + prediction) in frames."""
        return self.config.org_obs_len + self.config.org_pred_len

    @property
    def resampling_ratio(self) -> float:
        """Ratio of target sample time to original sample time."""
        return self.config.org_sample_time / self.config.target_sample_time

    @override
    def input_len(self) -> int:
        return int((self.config.org_obs_len - 1) * self.resampling_ratio + 1)

    @override
    def output_len(self) -> int:
        total_len = int((self.sequence_length - 1) * self.resampling_ratio + 1)
        return total_len - self.input_len()

    @override
    def sources(self) -> Iterable[tuple[str, pl.DataFrame]]:
        if not isinstance(self.config.dataset, set):
            datasets = {self.config.dataset}
        else:
            datasets = self.config.dataset
        for dataset in datasets:
            data_dir = self.config.data_root / dataset / self.config.split
            # Sort to ensure consistent order across runs and systems.
            for data_file in sorted(data_dir.iterdir()):
                yield (
                    data_file.name,
                    # The source is a loaded polars LazyFrame containing the xy
                    # trajectory data of all pedestrian for a certain scene.
                    self._read_data_file(data_file),
                )

    @override
    def load_raw(self, source: pl.DataFrame) -> Iterable[pl.DataFrame]:
        # For some data sources (files) there are some pedestrians with only
        # one frame of data, which is pointless and will cause issues during
        # resampling, so we filter them out here.
        source = source.filter(pl.col("frame").n_unique().over("id") > 1)
        max_frame = source.select(pl.col("frame").max()).item() + 1
        if max_frame is None:
            return

        # IMPORTANT: The data is already sorted by ["id", "frame"] due to the
        # way we read it, so we can just use group_by_dynamic to create the
        # sliding windows of frames.
        for _, window in source.group_by_dynamic(
            "frame",
            every=f"{self.config.window_skip}i",
            period=f"{self.sequence_length}i",
            include_boundaries=False,
        ):
            if not window.is_empty():
                yield window

    @override
    def normalize(self, df: pl.DataFrame) -> pl.DataFrame | None:
        # Input is a scene DataFrame of length (obs_len + pred_len) *
        # num_pedestrians, with columns "frame", "id", "x", "y", "vx", "vy".
        if df.is_empty():
            return None

        start = df.select(pl.col("frame").min()).item()
        pred_frame = start + self.config.org_obs_len - 1
        if df.filter(pl.col("frame") == pred_frame).is_empty():
            return None

        if self.config.require_all_valid:
            df = df.filter(pl.col("frame").len().over("id") == self.sequence_length)

        if (
            df.is_empty()
            or df.select(pl.col("id").n_unique()).item() < self.config.min_pedestrian
        ):
            return None

        df = resample_tracks(
            df,
            ratio=self.resampling_ratio,
            group_by="id",
            pos_columns=["x", "y"],
            add_velocity=True,
            add_acceleration=True,
        )

        return yaw_from_vel(df, yaw_col="yaw").with_columns(
            pl.lit(Category.PEDESTRIAN).alias("agent_class"),
        )

    def _read_data_file(self, path: Path) -> pl.DataFrame:
        return (
            pl.scan_csv(
                path,
                has_header=False,
                separator=self.config.sep,
                new_columns=["frame", "id", "x", "y"],
                schema={
                    "frame": pl.Int32,
                    "id": pl.Int32,
                    "x": pl.Float64,
                    "y": pl.Float64,
                },
            )
            .with_columns(
                ((pl.col("frame") - pl.col("frame").min()) // 10).cast(pl.Int32),
                pl.col("id").cast(pl.Int32),
            )
            .collect()
        )


if __name__ == "__main__":
    config = Config(
        data_root=Path("data"),
        dataset={"hotel"},
        split="test",
        org_sample_time=0.4,
        target_sample_time=0.1,
    )
    processor = EthUcyProcessor(config)
    count: int = 0
    total_time = 0.0
    for scene in processor.process_scenes():
        ...
