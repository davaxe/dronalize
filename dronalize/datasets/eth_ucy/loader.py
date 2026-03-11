from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.core import AgentCategory, BaseSceneLoader
from dronalize.core.interfaces import IngestOutput, Source
from dronalize.core.split import DatasetSplit
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class _EthUcyLoader(BaseSceneLoader[Path]):
    """Loader for ETH/UCY pedestrian trajectory datasets."""

    def __init__(
        self,
        data_root: Path | str,
        dataset: str | Sequence[str],
        *,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        split: DatasetSplit | None = None,
    ) -> None:
        """Initialize with the given configuration.

        Parameters
        ----------
        data_root : Path or str
            Path to the root directory containing the ETH/UCY data.
        dataset : str or Sequence[str]
            Name(s) of the dataset(s) to load (e.g., "hotel", "eth").
        loader_config : , optional
            Loader configuration override. If None, the default configuration is used.
        split : DatasetSplit, optional
            Which dataset split to load. Defaults to all sources.

        """
        super().__init__(loader_config=loader_config, map_config=map_config, split=split)
        self._data_root = self._normalize_data_root(data_root)
        self._dataset = {dataset} if isinstance(dataset, str) else set(dataset)

    def _sources_from_split(self, split_name: str) -> Iterable[Source[Path]]:
        for dataset in sorted(self._dataset):
            data_dir = self._data_root / dataset / split_name
            if not data_dir.is_dir():
                continue
            for data_file in sorted(data_dir.iterdir()):
                yield Source(identifier=data_file.name, inner=data_file)

    @override
    def all_sources(self) -> Iterable[Source[Path]]:
        yield from self.train_sources()
        yield from self.validate_sources()
        yield from self.test_sources()

    @override
    def train_sources(self) -> Iterable[Source[Path]]:
        return self._sources_from_split("train")

    @override
    def validate_sources(self) -> Iterable[Source[Path]]:
        return self._sources_from_split("val")

    @override
    def test_sources(self) -> Iterable[Source[Path]]:
        return self._sources_from_split("test")

    @override
    def ingest(self, source: Source[Path]) -> Iterable[IngestOutput]:
        yield (
            pl.scan_csv(
                source.inner,
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
            ),
            None,
        )

    @override
    def pipeline(self) -> Pipeline:
        config = self.loader_config
        return (
            Pipeline()
            .compose(trajectory_pipeline(config, derivative_rename=self.derivative_names()))
            .then(tr.yaw_from_vel())
            .then(tr.with_columns(agent_category=pl.lit(AgentCategory.PEDESTRIAN)))
        )

    @override
    def num_sources(self) -> int | None:
        num_sources: int = 0
        split = self._split

        split_names: list[str] = []
        if split in {DatasetSplit.ALL, DatasetSplit.TRAIN}:
            split_names.append("train")
        if split in {DatasetSplit.ALL, DatasetSplit.VAL}:
            split_names.append("val")
        if split in {DatasetSplit.ALL, DatasetSplit.TEST}:
            split_names.append("test")

        for dataset in self._dataset:
            for split_name in split_names:
                data_dir = self._data_root / dataset / split_name
                if data_dir.is_dir():
                    num_sources += sum(1 for _ in data_dir.iterdir())

        return num_sources

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(
                input_len=8,
                output_len=12,
                sample_time=0.4,
            )
            .with_window(step_size=1)
            .with_filtering(require_all_valid=True, min_samples_per_agent=2)
            .with_resampling(4, 1, method="fast")
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.no_map()


class HotelLoader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY hotel dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        *,
        split: DatasetSplit | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset="hotel",
            loader_config=loader_config,
            map_config=map_config,
            split=split,
        )


class EthLoader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY eth dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        *,
        split: DatasetSplit | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset="eth",
            loader_config=loader_config,
            map_config=map_config,
            split=split,
        )


class UnivLoader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY univ dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        *,
        split: DatasetSplit | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset="univ",
            loader_config=loader_config,
            map_config=map_config,
            split=split,
        )


class Zara1Loader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY zara1 dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        *,
        split: DatasetSplit | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset="zara1",
            loader_config=loader_config,
            map_config=map_config,
            split=split,
        )


class Zara2Loader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY zara2 dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        *,
        split: DatasetSplit | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset="zara2",
            loader_config=loader_config,
            map_config=map_config,
            split=split,
        )
