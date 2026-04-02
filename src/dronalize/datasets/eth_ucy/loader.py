"""Loader implementations for the ETH/UCY pedestrian datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

import dronalize.processing.pipeline.transforms as tr
from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_ONLY
from dronalize.processing.filters import Filter
from dronalize.processing.filters.agent import MinSamples
from dronalize.processing.ingest.base import BaseSceneLoader, LoaderSplitCapabilities
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.ingest.loader import IngestedData, Source
from dronalize.processing.pipeline.functional.resample import ResampleSpec
from dronalize.processing.pipeline.functional.resample._common import ResampleMethod

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.scene import SceneSchema
    from dronalize.processing.ingest.splits import SplitConfig
    from dronalize.processing.maps.config import MapConfig
    from dronalize.processing.pipeline.pipeline import Pipeline


class _EthUcyLoader(BaseSceneLoader):
    """Loader for ETH/UCY pedestrian trajectory datasets."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_source_split=True
    )

    def __init__(
        self,
        data_root: Path | str,
        *,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
    ) -> None:
        """Initialize the ETH/UCY loader.

        Parameters
        ----------
        data_root : Path or str
            Path to the root directory containing the ETH/UCY data.
        loader_config : LoaderConfig, optional
            Loader configuration override.
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
        return (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

    def _sources_from_split(self, split_name: str) -> Iterable[Source[Path]]:
        data_dir = self._data_root / split_name
        for data_file in sorted(data_dir.iterdir()):
            yield Source(identifier=data_file.name, data=data_file)

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[Path]]:
        return self._sources_from_split(split.value)

    @override
    def ingest(self, source: Source[Path]) -> Iterable[IngestedData]:
        yield IngestedData(
            pl.scan_csv(
                source.data,
                has_header=False,
                separator="\t",
                new_columns=["frame", "id", "x", "y"],
                schema=pl.Schema({
                    "frame": pl.Int32,
                    "id": pl.Int32,
                    "x": pl.Float64,
                    "y": pl.Float64,
                }),
            ).with_columns(
                ((pl.col("frame") - pl.col("frame").min()) // 10).cast(pl.Int32),
                pl.col("id").cast(pl.Int32),
            )
        )

    @override
    def pipeline(self) -> Pipeline:
        return (
            super()
            .pipeline()
            .then(tr.with_columns(agent_category=pl.lit(AgentCategory.PEDESTRIAN)))
        )

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_ONLY

    @override
    def num_sources(self) -> int | None:
        splits = self.splits if self.splits is not None else self.predefined_splits()
        return sum(self._count_sources_for_split(split) for split in splits)

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=8, output_len=12, sample_time=0.4)
            .with_window(step=1)
            .with_filter(Filter.define(agent_rules=[MinSamples(minimum=2)]))
            .with_resampling(ResampleSpec(up=4, down=1, method=ResampleMethod.LINEAR))
        )

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        num_sources = 0
        data_dir = self._data_root / split.value
        if data_dir.is_dir():
            num_sources += sum(1 for _ in data_dir.iterdir())
        return num_sources


class HotelLoader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY hotel dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )


class EthLoader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY eth dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )


class UnivLoader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY univ dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )


class Zara1Loader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY zara1 dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )


class Zara2Loader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY zara2 dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )


if __name__ == "__main__":
    import os

    from dronalize.datasets.eth_ucy import DESCRIPTORS
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env

    dataset_name = os.environ.get("ETH_UCY_DATASET", "hotel")
    descriptor = DESCRIPTORS[dataset_name]
    root = resolve_dataset_root_from_env(
        "ethucy", dataset_name, alternatives=[("eth_ucy", dataset_name)]
    )
    _ = debug_descriptor(descriptor, root)
