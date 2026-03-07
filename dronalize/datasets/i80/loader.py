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


class I80Loader(BaseSceneLoader[int, Path]):
    """Scene loader for the I-80 dataset."""

    def __init__(
        self,
        data_dir: Path,
        loader_config: LoaderConfig | None = None,
        lane_change_ratio: float | None = 1.0,
    ) -> None:
        """Initialize the I80 dataset loader.

        It is possible to rebalance the dataset by adjusting the number of lane
        changing agents compared to non-lane changing agents. This can be done
        by setting the `lane_change_ratio` parameter. For example, a ratio of
        0.5 would result in half as many lane changing agents as non-lane
        changing agents. Typically highway datasets are heavily imbalanced
        towards non-lane changing agents, which means that a high ratio con
        result in way less total data.

        Parameters
        ----------
        data_dir : Path
            Path to root of the I80 dataset, containing subdirectories of data files.
        loader_config : LoaderConfig, optional
            Processor configuration. If None, default configuration will be used.
        lane_change_ratio : float, optional
            Ratio for rebalancing highway agents. If None, no rebalancing will
            be applied. Default is 1.0, i.e. same number of lane changes as
            non-lane changes.

        """
        super().__init__(loader_config=loader_config, enforce_schema=False)
        self._data_dir = data_dir
        self._rebalance_ratio = lane_change_ratio

    @override
    def sources(self) -> Iterable[Source[int, Path]]:
        for i, csv_file in enumerate(self._data_dir.rglob("trajectories*.csv")):
            yield Source(identifier=i, inner=csv_file)

    @override
    def ingest(self, source: Source[int, Path]) -> Iterable[IngestOutput]:
        yield (
            pl.scan_csv(source.inner).select(
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
                self._lane_changes_expr(),
            ),
            None,
        )

    @override
    def num_sources(self) -> int | None:
        return sum(1 for _ in self._data_dir.rglob("trajectories*.csv"))

    @override
    def pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .then_if_present(
                tr.rebalance,
                arg=self._rebalance_ratio,
            )
            .compose(
                trajectory_pipeline(self.loader_config, derivative_rename=self.derivative_names())
            )
            .then(tr.yaw_from_vel())
        )

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(20, 50, 0.1).with_window(25).with_filtering(require_frames=[19])

    @staticmethod
    def _lane_changes_expr(
        lane_id_col: str = "Lane_ID",
        id_col: str = "Vehicle_ID",
    ) -> pl.Expr:
        return (
            pl
            .col(lane_id_col)
            .ne(pl.col(lane_id_col).shift())
            .fill_null(value=False)
            .sum()
            .over(id_col)
            .alias("lane_changes")
        )
