"""Loader implementation for the ApolloScape dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.errors import SplitNotSupportedError
from dronalize.core.scene import POSITIONS_YAW
from dronalize.processing.filtering.agent import MinSamples
from dronalize.processing.filtering.filter import Filter
from dronalize.processing.loading.base import BaseSceneLoader, LoaderSplitCapabilities
from dronalize.processing.loading.config import LoaderConfig
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.processing.pipeline.functional.resample import ResampleSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.loading.splits import SplitConfig
    from dronalize.processing.maps.config import MapConfig


class ApolloScapeLoader(BaseSceneLoader):
    """Loader for the ApolloScape dataset."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_source_split=True
    )

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
    ) -> None:
        """Initialize the ApolloScape loader.

        The raw split directories contain space-separated text files with the
        trajectory columns defined in `_DATA_SCHEMA`.

        Parameters
        ----------
        data_root : Path or str
            Root directory of the ApolloScape dataset. It should contain
            `prediction_train/`, `prediction_test/`, and `val_split/`.
        loader_config : LoaderConfig, optional
            Configuration override.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Optional selection of predefined dataset splits. `None` processes
            all available sources.
        """
        super().__init__(
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )
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
            yield Source(identifier=data_file.stem, data=data_file)

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[Path]]:
        if split is DatasetSplit.TRAIN:
            return self._sources_from_dir(self._data_root / "prediction_train")
        if split is DatasetSplit.VAL:
            return self._sources_from_dir(self._data_root / "val_split")
        raise SplitNotSupportedError(type(self).__name__, split)

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        yield LoadedSourceData(
            pl.scan_csv(source.data, has_header=False, schema=_DATA_SCHEMA, separator=" ").select(
                *("frame", "id", "x", "y", "yaw"),
                pl.col("agent_category").replace_strict({
                    1: AgentCategory.CAR.value,
                    2: AgentCategory.TRUCK.value,
                    3: AgentCategory.PEDESTRIAN.value,
                    4: AgentCategory.BICYCLE.value,
                    5: AgentCategory.UNKNOWN.value,
                }),
            )
        )

    @override
    def num_sources(self) -> int | None:
        splits: Iterable[DatasetSplit] = (
            self.splits if self.splits is not None else self.predefined_splits()
        )
        return sum(self._count_sources_for_split(split) for split in splits)

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_YAW

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=4, output_len=6, sample_time=0.5)
            .with_resampling(
                ResampleSpec
                .cubic(up=5, down=1)
                .with_output_derivative(1, "vx", "vy")
                .with_output_derivative(2, "ax", "ay")
            )
            .with_filter(Filter.define_cleanup(MinSamples(minimum=2)))
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


if __name__ == "__main__":
    from dronalize.datasets.apolloscape import DATASET_SPEC
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env

    root = resolve_dataset_root_from_env("apollo")
    scenes = debug_descriptor(DATASET_SPEC, root, step=150)
    for scene in scenes:
        scene.frame.write_csv(f"debug_scene_{scene.scene_number}.csv")
