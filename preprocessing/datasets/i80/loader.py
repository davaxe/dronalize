from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from preprocessing.common.trajectory_utils.basic import (
    yaw_from_vel,
)
from preprocessing.common.trajectory_utils.filter import rebalance_highway_agents
from preprocessing.common.trajectory_utils.process import prepare_agent_trajectories
from preprocessing.core import AgentCategory
from preprocessing.core.interface import BaseSceneLoader, LoaderConfig

if TYPE_CHECKING:
    from collections.abc import Iterable


class I80Loader(BaseSceneLoader[int, pl.LazyFrame]):
    """Scene loader for the I-80 dataset."""

    def __init__(
        self,
        data_dir: Path,
        config: LoaderConfig | None = None,
        lane_change_ratio: float | None = 1.0,
    ) -> None:
        """Initialize the I80 dataset loader.

        It is possible to rebalance the dataset by adjusting the number of lane changing agents
        compared to non-lane changing agents. This can be done by setting the `lane_change_ratio`
        parameter. For example, a ratio of 0.5 would result in half as many lane changing agents as
        non-lane changing agents. Typically highway datasets are heavily imbalanced towards non-lane
        changing agents, which means that a high ratio con result in way less total data.

        Args:
            data_dir: Path to root of the I80 dataset, containing subdirectores of data files.
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
        for i, csv_file in enumerate(self._data_dir.rglob("trajectories*.csv")):
            yield (
                i,
                pl.scan_csv(csv_file).select(
                    pl.col("Vehicle_ID").alias("id"),
                    pl.col("Frame_ID").alias("frame"),
                    pl.col("Local_X").alias("x"),
                    pl.col("Local_Y").alias("y"),
                    pl
                    .col("v_Class")
                    .replace_strict({
                        1: AgentCategory.MOTORCYCLE,
                        2: AgentCategory.CAR,
                        3: AgentCategory.TRUCK,
                    })
                    .alias("agent_category"),
                    "Lane_ID",
                    self._lane_changes_expr(),
                ),
            )

    @staticmethod
    def _lane_changes_expr(lane_id_col: str = "Lane_ID", id_col: str = "Vehicle_ID") -> pl.Expr:
        return (
            pl
            .col(lane_id_col)
            .ne(pl.col(lane_id_col).shift())  # Compares the current lane to the previous lane
            .fill_null(value=False)  # Prevents the first row of each ID from counting as a change
            .sum()  # Adds up all the True values (the actual lane changes)
            .over(id_col)  # Applies the logic independently for each unique "id"
            .alias("lane_changes")
        )

    @override
    def load_raw(self, source: pl.LazyFrame) -> Iterable[pl.LazyFrame]:
        yield from prepare_agent_trajectories(
            rebalance_highway_agents(source, ratio=self._rebalance_ratio).drop("lane_changes")
            if self._rebalance_ratio
            else source.drop("lane_changes"),
            self.processor_config,
            add_derivative=True,
            add_second_derivative=True,
            derivative_rename=self.derivative_names(),
        )

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return yaw_from_vel(df)

    @override
    def default_config(self) -> LoaderConfig:
        return LoaderConfig(20, 50, 0.1).window_parameters(25).scene_filtering_parameters()


if __name__ == "__main__":
    loader = I80Loader(Path("/home/west/Developer/behavior-prediction/datasets/i80/"))
    for scene in loader.scenes():
        print(scene)
