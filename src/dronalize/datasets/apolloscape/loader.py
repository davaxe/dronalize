from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config.loader import LoaderConfig
from dronalize.exceptions import SplitNotSupportedError
from dronalize.loading import BaseSceneLoader
from dronalize.loading.loader import IngestOutput, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline
from dronalize.scene import POSITIONS_YAW_V1

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.config.map import MapConfig
    from dronalize.scene import SceneSchema


class ApolloScapeLoader(BaseSceneLoader[Path]):
    """Loader for the ApolloScape dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
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
        data_root : Path or str
            The root directory of the ApolloScape dataset.  This directory
            should contain `prediction_train/`, `prediction_test/`,
            and `val_split/` subdirectories.
        loader_config : LoaderConfig, optional
            Configuration override.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. Can contain one or more predefined splits,
            or `None` to process all sources.

        """
        super().__init__(loader_config=loader_config, map_config=map_config, splits=splits)
        self._data_root: Path = Path(data_root)

    @classmethod
    @override
    def predefined_splits(cls) -> tuple[DatasetSplit, ...]:
        return (DatasetSplit.TRAIN, DatasetSplit.VAL)

    @staticmethod
    def _sources_from_dir(data_dir: Path) -> Iterable[Source[Path]]:
        if not data_dir.is_dir():
            return
        for data_file in sorted(data_dir.glob("*.txt")):
            yield Source(identifier=data_file.stem, inner=data_file)

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[Path]]:
        if split is DatasetSplit.TRAIN:
            return self._sources_from_dir(self._data_root / "prediction_train")
        if split is DatasetSplit.VAL:
            return self._sources_from_dir(self._data_root / "val_split")
        raise SplitNotSupportedError(type(self).__name__, split)

    @override
    def ingest(self, source: Source[Path]) -> Iterable[IngestOutput]:
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
        splits: Iterable[DatasetSplit] = (
            self.splits if self.splits is not None else self.predefined_splits()
        )
        return sum(self._count_sources_for_split(split) for split in splits)

    @override
    def pipeline(self) -> Pipeline:
        return Pipeline().compose(trajectory_pipeline(self.loader_config))

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_YAW_V1

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=4, output_len=6, sample_time=0.5)
            .with_resampling(5, 1)
            .with_filtering(require_frames=[3])
            .with_window(1)
        )

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        if split is DatasetSplit.TRAIN:
            return sum(1 for _ in (self._data_root / "prediction_train").glob("*.txt"))
        if split is DatasetSplit.VAL:
            return sum(1 for _ in (self._data_root / "val_split").glob("*.txt"))
        raise SplitNotSupportedError(type(self).__name__, split)


_DATA_SCHEMA: pl.Schema = pl.Schema({
    "frame": pl.Int64,
    "id": pl.Int64,
    "agent_category": pl.Int64,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "length": pl.Float64,
    "width": pl.Float64,
    "height": pl.Float64,
    "yaw": pl.Float64,
})
