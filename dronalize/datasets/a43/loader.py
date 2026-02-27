from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.common.trajectory.basic import yaw_from_vel
from dronalize.common.trajectory.process import prepare_agent_trajectories
from dronalize.core import AgentCategory, BaseSceneLoader, LoaderConfig
from dronalize.core.datatypes import map_context as mc
from dronalize.core.protocols.loader import Source

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class A43Loader(BaseSceneLoader[int, pl.LazyFrame]):
    """Scene loader for the A43 dataset."""

    def __init__(
        self,
        data_dir: Path,
        config: LoaderConfig | None = None,
    ) -> None:
        """Initialize the A43 dataset loader.

        Parameters
        ----------
        data_dir : Path
            Path to root of the A43 dataset, data files.
        config : LoaderConfig, optional
            Processor configuration. If None, default configuration will be used.

        """
        super().__init__(loader_config=config, enforce_schema=False)
        self._data_dir = data_dir

    @override
    def sources(self) -> Iterable[Source[int, pl.LazyFrame]]:
        for i, csv_file in enumerate(self._data_dir.glob("*.csv")):
            yield Source(
                identifier=i,
                inner=pl.scan_csv(csv_file).select(
                    pl.col("ID").alias("id"),
                    pl.col("tseconds").rank("dense").sub(1).alias("frame").cast(pl.Int64),
                    *("x", "y", "vy", "vx", "ax", "ay"),
                    pl
                    .col("VehicleCategory")
                    .replace_strict({
                        "Motorcycle": AgentCategory.MOTORCYCLE,
                        "Passenger Car": AgentCategory.CAR,
                        "Semi-trailer truck": AgentCategory.TRUCK,
                        "Truck": AgentCategory.TRUCK,
                        "Van": AgentCategory.VAN,
                        "Bus": AgentCategory.BUS,
                    })
                    .alias("agent_category"),
                ),
            )

    @override
    def num_sources(self) -> int | None:
        return sum(1 for _ in self._data_dir.rglob("trajectories*.csv"))

    @override
    def load_raw(
        self, source: Source[int, pl.LazyFrame]
    ) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
        for df in prepare_agent_trajectories(
            source.inner,
            self.loader_config,
            add_derivative=True,
            add_second_derivative=True,
            derivative_rename=self.derivative_names(),
        ):
            yield df, mc.NoMap()

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return yaw_from_vel(df)

    @override
    def default_config(self) -> LoaderConfig:
        return LoaderConfig(20, 50, 0.1).window_parameters(25).scene_filtering_parameters()
