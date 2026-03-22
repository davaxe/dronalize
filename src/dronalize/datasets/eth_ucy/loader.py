from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.loading import BaseSceneLoader, IngestOutput, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.functional.resample import ResampleSpec
from dronalize.pipeline.functional.resample._common import ResampleMethod
from dronalize.scene import POSITIONS_ONLY_V1

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.pipeline.pipeline import Pipeline
    from dronalize.scene import SceneSchema


class _EthUcyLoader(BaseSceneLoader[Path]):
    """Loader for ETH/UCY pedestrian trajectory datasets."""

    def __init__(
        self,
        data_root: Path | str,
        *,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        """Initialize with the given configuration.

        Parameters
        ----------
        data_root : Path or str
            Path to the root directory containing the ETH/UCY data.
        loader_config : , optional
            Loader configuration override. If None, the default configuration is used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. Can contain one or more predefined splits,
            or `None` to process all sources.

        """
        super().__init__(loader_config=loader_config, map_config=map_config, splits=splits)
        self._data_root: Path = Path(data_root)

    @classmethod
    @override
    def predefined_splits(cls) -> tuple[DatasetSplit, ...]:
        return (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

    def _sources_from_split(self, split_name: str) -> Iterable[Source[Path]]:
        data_dir = self._data_root / split_name
        for data_file in sorted(data_dir.iterdir()):
            yield Source(identifier=data_file.name, inner=data_file)

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[Path]]:
        return self._sources_from_split(split.value)

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
        return trajectory_pipeline(config).then(
            tr.with_columns(agent_category=pl.lit(AgentCategory.PEDESTRIAN))
        )

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_ONLY_V1

    @override
    def num_sources(self) -> int | None:
        splits = self.splits if self.splits is not None else self.predefined_splits()
        return sum(self._count_sources_for_split(split) for split in splits)

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
            .with_filtering(require_all_valid=True, min_samples_per_agent=2, require_frames=None)
            .with_resampling(
                ResampleSpec(
                    up=4,
                    down=1,
                    method=ResampleMethod.PCHIP,
                    output_derivatives={1: {"vx": None, "vy": None}, 2: {"ax": None, "ay": None}},
                ),
            )
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.no_map()

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
    ) -> None:
        super().__init__(
            data_root=data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
        )


class EthLoader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY eth dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
        )


class UnivLoader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY univ dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
        )


class Zara1Loader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY zara1 dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
        )


class Zara2Loader(_EthUcyLoader):
    """Convenience alias for the ETH/UCY zara2 dataset."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
        )
