from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.common.trajectory.process import prepare_agent_trajectories
from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.datatypes.loader_config import LoaderConfig
from dronalize.core.protocols.loader import BaseSceneLoader, Source

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class ApolloScapeLoader(BaseSceneLoader[str, pl.LazyFrame]):
    """Loader for the ApolloScape dataset."""

    def __init__(self, data_dir: Path, loader_config: LoaderConfig | None = None) -> None:
        """Initialize the loader with the given data directory and loader configuration.

        The `data_dir` directory should contain the actual .txt datafiles stored
        in CSV-like format, with the following columns (in order):
        - frame: The frame number of the trajectory point.
        - id: The unique identifier for the agent.
        - agent_category: The category of the agent, encoded as an integer.
        - x: The x-coordinate of the agent's position.
        - y: The y-coordinate of the agent's position.
        - z: The z-coordinate of the agent's position.
        - length: The length of the agent.
        - width: The width of the agent.
        - height: The height of the agent.
        - yaw: The yaw angle of the agent in radians.

        Parameters
        ----------
        data_dir : Path
            The directory containing the ApolloScape data files.
        loader_config : LoaderConfig, optional
            Configuration override.

        """
        super().__init__(loader_config, enforce_schema=True)
        self._data_dir = data_dir

    @override
    def sources(self) -> Iterable[Source[str, pl.LazyFrame]]:
        for data_file in self._data_dir.glob("*.txt"):
            yield Source(
                identifier=data_file.stem,
                inner=pl.scan_csv(
                    data_file,
                    has_header=False,
                    schema=_DATA_SCHEMA,
                    separator=" ",
                ).select(
                    *("frame", "id", "x", "y", "yaw"),
                    pl.col("agent_category").replace_strict({
                        1: AgentCategory.CAR.value,
                        2: AgentCategory.TRUCK.value,
                        3: AgentCategory.PEDESTRIAN.value,
                        4: AgentCategory.BICYCLE.value,
                        5: AgentCategory.UNKNOWN.value,
                    }),
                ),
            )

    @override
    def num_sources(self) -> int | None:
        return sum(1 for _ in self._data_dir.glob("*.txt"))

    @override
    def load_raw(
        self,
        source: Source[str, pl.LazyFrame],
    ) -> Iterable[tuple[pl.LazyFrame, None]]:
        for df in prepare_agent_trajectories(
            source.inner,
            self.loader_config,
            add_derivative=True,
            add_second_derivative=True,
            derivative_rename=self.derivative_names(),
        ):
            yield df, None

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(4, 6, 0.5)
            .with_resampling(5, 1)
            .with_filtering(require_frames=[3])
            .with_window(1)
        )


_DATA_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int64,
    "id": pl.Int64,
    "agent_category": pl.Int64,
    "x": pl.Float32,
    "y": pl.Float32,
    "z": pl.Float32,
    "length": pl.Float32,
    "width": pl.Float32,
    "height": pl.Float32,
    "yaw": pl.Float32,
})

if __name__ == "__main__":
    import os
    import time
    from pathlib import Path

    # Get root from env-var
    root = Path(os.getenv("TRAJ_DATA", "")) / "apollo" / "prediction_train"
    loader = ApolloScapeLoader(root)
    time_start = time.perf_counter()
    counter = 0
    for _ in loader.scenes():
        counter += 1
        if counter % 1 == 0:
            print(f"Loaded {counter} scenes in {time.perf_counter() - time_start:.2f} seconds")
    print(f"Total {counter} scenes in {time.perf_counter() - time_start:.2f} seconds")
