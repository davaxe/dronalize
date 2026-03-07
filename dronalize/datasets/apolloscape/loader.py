from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.datatypes.loader_config import LoaderConfig
from dronalize.core.datatypes.split import DatasetSplit
from dronalize.core.pipeline import Pipeline
from dronalize.core.pipelines_factories import trajectory_pipeline
from dronalize.core.protocols.loader import BaseSceneLoader, IngestOutput, Source

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class ApolloScapeLoader(BaseSceneLoader[str, Path]):
    """Loader for the ApolloScape dataset."""

    def __init__(
        self,
        data_root: Path,
        loader_config: LoaderConfig | None = None,
        *,
        split: DatasetSplit = DatasetSplit.ALL,
    ) -> None:
        """Initialize the loader with the given data directory and loader configuration.

        Each split subdirectory should contain the actual .txt datafiles stored
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
        data_root : Path
            The root directory of the ApolloScape dataset.  This directory
            should contain `prediction_train/`, `prediction_test/`,
            and `val_split/` subdirectories.
        loader_config : LoaderConfig, optional
            Configuration override.
        split : DatasetSplit, optional
            Which dataset split to load.  Defaults to `DatasetSplit.ALL`.

        """
        super().__init__(loader_config, enforce_schema=True, split=split)
        self._data_root = data_root

    # ------------------------------------------------------------------
    # Split-aware source discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _sources_from_dir(data_dir: Path) -> Iterable[Source[str, Path]]:
        if not data_dir.is_dir():
            return
        for data_file in sorted(data_dir.glob("*.txt")):
            yield Source(identifier=data_file.stem, inner=data_file)

    @override
    def all_sources(self) -> Iterable[Source[str, Path]]:
        yield from self.train_sources()
        yield from self.validate_sources()
        yield from self.test_sources()

    @override
    def train_sources(self) -> Iterable[Source[str, Path]]:
        return self._sources_from_dir(self._data_root / "prediction_train")

    @override
    def validate_sources(self) -> Iterable[Source[str, Path]]:
        return self._sources_from_dir(self._data_root / "val_split")

    @override
    def test_sources(self) -> Iterable[Source[str, Path]]:
        return self._sources_from_dir(self._data_root / "prediction_test")

    # ------------------------------------------------------------------
    # Ingestion / pipeline
    # ------------------------------------------------------------------

    @override
    def ingest(self, source: Source[str, Path]) -> Iterable[IngestOutput]:
        yield (
            pl.scan_csv(
                source.inner,
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
            None,
        )

    @override
    def num_sources(self) -> int | None:
        dirs: list[Path] = []
        split = self._split
        if split in {DatasetSplit.ALL, DatasetSplit.TRAIN}:
            dirs.append(self._data_root / "prediction_train")
        if split in {DatasetSplit.ALL, DatasetSplit.VALIDATE}:
            dirs.append(self._data_root / "val_split")
        if split in {DatasetSplit.ALL, DatasetSplit.TEST}:
            dirs.append(self._data_root / "prediction_test")

        return sum(sum(1 for _ in d.glob("*.txt")) for d in dirs if d.is_dir())

    @override
    def pipeline(self) -> Pipeline:
        return Pipeline().compose(
            trajectory_pipeline(self.loader_config, derivative_rename=self.derivative_names())
        )

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
    root = Path(os.getenv("TRAJ_DATA", "")) / "apollo"
    loader = ApolloScapeLoader(root, split=DatasetSplit.TRAIN)
    time_start = time.perf_counter()
    counter = 0
    for _ in loader.scenes():
        counter += 1
        if counter % 1 == 0:
            print(f"Loaded {counter} scenes in {time.perf_counter() - time_start:.2f} seconds")
    print(f"Total {counter} scenes in {time.perf_counter() - time_start:.2f} seconds")
