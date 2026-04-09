"""Loader implementation for the I-80 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_ONLY
from dronalize.datasets.shared import utils
from dronalize.processing.filtering import Filter
from dronalize.processing.filtering.agent import MinSamples
from dronalize.processing.loading.base import BaseSceneLoader, LoaderSplitCapabilities
from dronalize.processing.loading.config import LoaderConfig
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.processing.maps.config import MapConfig
from dronalize.processing.maps.resolver import MapResolver, no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.loading.splits import SplitConfig


class I80Loader(BaseSceneLoader):
    """Scene loader for the I-80 dataset."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_block_split=True
    )

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
    ) -> None:
        """Initialize the I80 dataset loader.

        Parameters
        ----------
        data_root : Path or str
            Root directory containing the extracted I-80 trajectory files.
        loader_config : LoaderConfig, optional
            Loader configuration override.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )
        self._data_root: Path = Path(data_root)

    @override
    def discover_sources(self) -> Iterable[Source[Path]]:
        for i, csv_file in enumerate(sorted(self._data_root.rglob("trajectories*.csv"))):
            yield Source(identifier=i, data=csv_file)

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        yield LoadedSourceData(
            pl.scan_csv(source.data).select(
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
                pl.col("Lane_ID").alias("lane_id"),
            )
        )

    @override
    def num_sources(self) -> int | None:
        return sum(1 for path in self._data_root.rglob("trajectories*.csv") if path.is_file())

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=20, output_len=50, sample_time=0.1)
            .with_window(25)
            .with_filter(Filter.define_cleanup(MinSamples(minimum=2)))
            .with_lane_change_sampling(required_lane_changes=3, negative_keep_every=3)
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.default()

    @override
    def map_resolver(self) -> MapResolver:
        if self._shared_memory_name is None or self.map_config is None:
            return no_map()
        return shared_map(self._shared_memory_name, utils.extract_fn(self.map_config.extraction))


if __name__ == "__main__":
    from dronalize.datasets.i80 import DATASET_SPEC
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env

    root = resolve_dataset_root_from_env("i80")
    _ = debug_descriptor(DATASET_SPEC, root)
