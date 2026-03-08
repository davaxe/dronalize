from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.core import AgentCategory, BaseSceneLoader, LoaderConfig
from dronalize.core.protocols.loader import IngestOutput, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable


class A43Loader(BaseSceneLoader[int, Path]):
    """Scene loader for the A43 dataset."""

    def __init__(
        self,
        data_root: Path,
        loader_config: LoaderConfig | None = None,
    ) -> None:
        """Initialize the A43 dataset loader.

        Parameters
        ----------
        data_root : Path
            Path to root of the A43 dataset, data files.
        loader_config : LoaderConfig, optional
            Processor configuration. If None, default configuration will be used.

        """
        super().__init__(loader_config=loader_config, enforce_schema=True)
        self._data_dir = data_root

    @override
    def all_sources(self) -> Iterable[Source[int, Path]]:
        for i, csv_file in enumerate(self._data_dir.glob("*.csv")):
            yield Source(identifier=i, inner=csv_file)

    @override
    def ingest(self, source: Source[int, Path]) -> Iterable[IngestOutput]:
        yield (
            pl.scan_csv(source.inner).select(
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
            None,
        )

    @override
    def num_sources(self) -> int | None:
        return sum(1 for _ in self._data_dir.rglob("trajectories*.csv"))

    @override
    def pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .compose(
                trajectory_pipeline(self.loader_config, derivative_rename=self.derivative_names())
            )
            .then(tr.yaw_from_vel())
        )

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
