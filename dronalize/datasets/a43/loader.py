from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.common.trajectory.basic import yaw_from_vel
from dronalize.common.trajectory.process import prepare_agent_trajectories
from dronalize.core import AgentCategory, BaseSceneLoader, LoaderConfig
from dronalize.core.pipeline import Pipeline
from dronalize.core.protocols.loader import Source

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class A43Loader(BaseSceneLoader[int, pl.LazyFrame]):
    """Scene loader for the A43 dataset."""

    def __init__(
        self,
        data_dir: Path,
        loader_config: LoaderConfig | None = None,
    ) -> None:
        """Initialize the A43 dataset loader.

        Parameters
        ----------
        data_dir : Path
            Path to root of the A43 dataset, data files.
        loader_config : LoaderConfig, optional
            Processor configuration. If None, default configuration will be used.

        """
        super().__init__(loader_config=loader_config, enforce_schema=True)
        self._data_dir = data_dir

    @override
    def sources(self) -> Iterable[Source[int, pl.LazyFrame]]:
        for i, csv_file in enumerate(self._data_dir.glob("*.csv")):
            yield Source(
                identifier=i,
                inner=pl.scan_csv(csv_file).select(
                    pl.col("ID").alias("id"),
                    pl.col("tseconds").round(1).rank("dense").sub(1).alias("frame").cast(pl.Int64),
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
    def ingest(self, source: Source[int, pl.LazyFrame]) -> Iterable[pl.LazyFrame]:
        # This should eventually do what `sources` do now, e.g. scan_csv (change
        # source to contain Path)
        yield source.inner

    @override
    def pipeline(self) -> Pipeline: ...

    @override
    def num_sources(self) -> int | None:
        return sum(1 for _ in self._data_dir.rglob("trajectories*.csv"))

    @override
    def load_raw(
        self,
        source: Source[int, pl.LazyFrame],
    ) -> Iterable[tuple[pl.LazyFrame, None]]:
        for df in prepare_agent_trajectories(
            source.inner,
            self.loader_config,
            add_derivative=True,
            add_second_derivative=True,
            derivative_rename=self.derivative_names(),
            stream_windows=False,
        ):
            yield df, None

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return yaw_from_vel(df)

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(20, 50, 0.1).with_window(25).with_filtering(require_frames=[19])


if __name__ == "__main__":
    import os
    import time
    from pathlib import Path

    # Get root from env-var
    root = Path(os.getenv("TRAJ_DATA", "")) / "A43"
    loader = A43Loader(root)
    time_start = time.perf_counter()
    counter = 0
    for _ in loader.scenes():
        counter += 1
        if counter % 1 == 0:
            print(f"Loaded {counter} scenes in {time.perf_counter() - time_start:.2f} seconds")
    print(f"Loaded {counter} scenes in {time.perf_counter() - time_start:.2f} seconds")
