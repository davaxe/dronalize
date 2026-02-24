from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from preprocessing.common.trajectory_utils.basic import (
    yaw_from_vel,
)
from preprocessing.common.trajectory_utils.filter import rebalance_highway_agents
from preprocessing.common.trajectory_utils.plot import plot_trajectories
from preprocessing.common.trajectory_utils.process import prepare_agent_trajectories
from preprocessing.core import AgentCategory
from preprocessing.core.interface import DataProcessor, ProcessorConfig

if TYPE_CHECKING:
    from collections.abc import Iterable


class HighDProcessor(DataProcessor[int, pl.LazyFrame]):
    """Processor for the highD dataset."""

    def __init__(
        self,
        data_dir: Path,
        config: ProcessorConfig | None = None,
        lane_change_ratio: float | None = 1.0,
    ) -> None:
        """Initialize the highD data processor.

        It is possible to rebalance the dataset by adjusting the number of lane changing agents
        compared to non-lane changing agents. This can be done by setting the `lane_change_ratio`
        parameter. For example, a ratio of 0.5 would result in half as many lane changing agents as
        non-lane changing agents. Typically highway datasets are heavily imbalanced towards non-lane
        changing agents, which means that a high ratio con result in way less total data.

        Args:
            data_dir: Path to the directory containing the highD dataset files.
            config: Optional processor configuration. If None, default configuration will be used.
            lane_change_ratio: Optional ratio for rebalancing highway agents. If None, no
                rebalancing will be applied. Default is 1.0, i.e. same number of lane changes as
                non-lane changes.

        """
        super().__init__(config, enforce_schema=False)
        self._data_dir = data_dir
        self._rebalance_ratio = lane_change_ratio

    @override
    def sources(self) -> Iterable[tuple[int, pl.LazyFrame]]:
        num_files: int = sum(1 for p in self._data_dir.iterdir() if p.is_file())
        for i in range(1, num_files // 4):
            meta = self._data_dir / f"{i:0>2}_tracksMeta.csv"
            tracks = self._data_dir / f"{i:0>2}_tracks.csv"
            meta_df = pl.scan_csv(meta, schema_overrides=_META_SCHEMA).select(
                "id",
                pl.col("numLaneChanges").alias("lane_changes"),
                pl
                .when(pl.col("class") == "Car")
                .then(AgentCategory.CAR)
                .otherwise(AgentCategory.TRUCK)
                .alias("agent_category"),
            )
            tracks_df = pl.scan_csv(tracks, schema_overrides=_TRACK_SCHEMA).select(
                "frame",
                "id",
                pl.col("x").add(pl.col("width") / 2),
                pl.col("y").add(pl.col("height") / 2),
                pl.col("xVelocity").alias("vx"),
                pl.col("yVelocity").alias("vy"),
                pl.col("xAcceleration").alias("ax"),
                pl.col("yAcceleration").alias("ay"),
            )
            combined = tracks_df.join(meta_df, left_on="id", right_on="id")
            yield i, combined

    @override
    def load_raw(self, source: pl.LazyFrame) -> Iterable[pl.LazyFrame]:
        yield from prepare_agent_trajectories(
            rebalance_highway_agents(source, ratio=self._rebalance_ratio).drop("lane_changes")
            if self._rebalance_ratio
            else source.drop("lane_changes"),
            self.processor_config,
        )

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        print(df.collect())
        chart = plot_trajectories(
            df.collect(),
            n_groups=None,
            group_by="id",
            highlight_frame=self.input_len,
            width=600,
            height=400,
        )
        chart.show()
        return yaw_from_vel(df)

    @override
    def default_config(self) -> ProcessorConfig:
        return (
            ProcessorConfig(50, 125, 0.04)
            .resampling_parameters(2, 5)
            .window_parameters(25)
            .scene_filtering_parameters()
        )


_META_SCHEMA: pl.Schema = pl.Schema({
    "id": pl.Int32,
    "numLaneChanges": pl.Int8,
})

_TRACK_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int32,
    "id": pl.Int32,
    "width": pl.Float32,
    "height": pl.Float32,
    "x": pl.Float32,
    "y": pl.Float32,
    "xVelocity": pl.Float32,
    "yVelocity": pl.Float32,
    "xAcceleration": pl.Float32,
    "yAcceleration": pl.Float32,
})

if __name__ == "__main__":
    data_dir = Path("/home/west/Developer/behavior-prediction/datasets/highD/data")

    processor = HighDProcessor(data_dir=data_dir)
    count = 0
    for scene in processor.scenes_iter():
        if count % 200 == 0:
            print(scene.inner)
            print(f"Processed {count} scenes")
        count += 1
    print(f"Total scenes processed: {count}")
