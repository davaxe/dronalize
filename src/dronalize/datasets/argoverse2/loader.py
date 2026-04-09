"""Loader implementation for the Argoverse 2 dataset."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from pydantic import Field
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_VELOCITY_YAW
from dronalize.datasets.argoverse2.maps.builder import Argoverse2MapBuilder
from dronalize.datasets.shared import utils
from dronalize.processing.filtering import Filter
from dronalize.processing.filtering.agent import RequireFrames
from dronalize.processing.filtering.cleanup import ExcludeCategories
from dronalize.processing.loading.base import (
    BaseSceneLoader,
    LoaderOptions,
    LoaderSplitCapabilities,
)
from dronalize.processing.loading.config import LoaderConfig
from dronalize.processing.loading.loader import LoadedSourceData, MapBinding, Source
from dronalize.processing.maps.config import MapConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.maps.graph import MapGraph
    from dronalize.core.scene import Scene, TrajectorySchema
    from dronalize.processing.loading.splits import SplitConfig
    from dronalize.processing.maps.resolver import MapResolver


class Argoverse2LoaderOptions(LoaderOptions):
    """Loader options for the Argoverse 2 dataset."""

    file_batch_size: int = Field(default=100, ge=1)


class Argoverse2Loader(BaseSceneLoader[list[Path], Argoverse2LoaderOptions]):
    """Loader for Argoverse 2 trajectory data stored in Parquet files."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_scene_split=True
    )

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
        *,
        loader_options: Argoverse2LoaderOptions | None = None,
    ) -> None:
        """Initialize the Argoverse 2 loader.

        Parameters
        ----------
        data_root : Path or str
            Root directory of the Argoverse 2 dataset. It should contain
            `train/`, `val/`, and `test/` subdirectories with scenario
            Parquet files.
        loader_options : Argoverse2LoaderOptions, optional
            Dataset-specific loader options. `file_batch_size` controls how
            many scenario files are processed in each batch.
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
            loader_options=loader_options,
        )
        self._data_root: Path = Path(data_root)
        self._file_batch_size: int = self.loader_options.file_batch_size

    @classmethod
    @override
    def loader_options_model(cls) -> type[Argoverse2LoaderOptions]:
        return Argoverse2LoaderOptions

    @classmethod
    @override
    def predefined_splits(cls) -> tuple[DatasetSplit, ...]:
        return (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

    def _sources_from_dir(self, data_dir: Path) -> Iterable[Source[list[Path]]]:
        if not data_dir.is_dir():
            return
        parquet_files = sorted(data_dir.glob("*/*.parquet"))
        for i in range(0, len(parquet_files), self._file_batch_size):
            batch_files = parquet_files[i : i + self._file_batch_size]
            yield Source(identifier=i, data=batch_files)

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[list[Path]]]:
        if split is DatasetSplit.TRAIN:
            return self._sources_from_dir(self._data_root / "train")
        if split is DatasetSplit.VAL:
            return self._sources_from_dir(self._data_root / "val")
        return self._sources_from_dir(self._data_root / "test")

    @override
    def load_source(self, source: Source[list[Path]]) -> Iterable[LoadedSourceData]:
        # Build a mapping from parquet file path to its co-located JSON map path.
        # Each parquet lives in a subdirectory like <scenario_id>/<scenario_id>.parquet
        # alongside a <scenario_id>.json map file.
        file_to_map: dict[str, str] = {}
        for pq in source.data:
            json_candidates = list(pq.parent.glob("*.json"))
            if json_candidates:
                file_to_map[str(pq)] = str(json_candidates[0])

        batch_lf = pl.scan_parquet(source.data, include_file_paths="file_id").select(
            pl.col("file_id"),
            self._map_object_type_expr("object_type").alias("agent_category"),
            pl.col("track_id").str.replace("AV", "0").cast(pl.Int32).alias("id"),
            pl.col("timestep").alias("frame"),
            pl.col("position_x").alias("x"),
            pl.col("position_y").alias("y"),
            pl.col("velocity_x").alias("vx"),
            pl.col("velocity_y").alias("vy"),
            pl.col("heading").alias("yaw"),
        )

        for (file_id,), group in batch_lf.collect().group_by(["file_id"]):
            map_path = file_to_map.get(str(file_id))
            yield LoadedSourceData(
                frame=group.lazy().drop("file_id"), map_binding=MapBinding(map_key=map_path)
            )

    @override
    def num_sources(self) -> int | None:
        splits = self.splits if self.splits is not None else self.predefined_splits()
        return sum(self._count_sources_for_split(split) for split in splits)

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_YAW

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(input_len=50, output_len=60, sample_time=0.1).with_filter(
            Filter.define(
                cleanup_rules=[
                    ExcludeCategories.define(
                        categories=[
                            AgentCategory.STATIC_OBJECT,
                            AgentCategory.UNKNOWN,
                            AgentCategory.UNIMPORTANT,
                        ]
                    )
                ],
                agent_rules=[RequireFrames.define(frames=[49])],
            )
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.full_map()

    @override
    def map_resolver(self) -> MapResolver:
        def _resolver(scene: Scene) -> MapGraph | None:
            if scene.map_key is None or self.map_config is None:
                return None

            return utils.extract_based_on_scene(
                self._get_map(
                    scene.map_key, self.map_config.min_distance, self.map_config.interp_distance
                ),
                scene,
                self.map_config.extraction,
            )

        return _resolver

    @staticmethod
    @functools.lru_cache(maxsize=10)
    def _get_map(key: str, min_distance: float | None, interp_distance: float | None) -> MapGraph:
        return Argoverse2MapBuilder.from_json_file(Path(key)).build(min_distance, interp_distance)

    @staticmethod
    def _map_object_type_expr(col: str) -> pl.Expr:
        mapping = {
            "static": AgentCategory.STATIC_OBJECT,
            "riderless_bicycle": AgentCategory.STATIC_OBJECT,
            "construction": AgentCategory.STATIC_OBJECT,
            "vehicle": AgentCategory.CAR,
            "motorcyclist": AgentCategory.MOTORCYCLE,
            "cyclist": AgentCategory.BICYCLE,
            "bus": AgentCategory.BUS,
            "pedestrian": AgentCategory.PEDESTRIAN,
            "background": AgentCategory.UNIMPORTANT,
            "unknown": AgentCategory.UNKNOWN,
        }
        return pl.col(col).replace_strict(
            mapping, default=AgentCategory.UNKNOWN, return_dtype=pl.Int32
        )

    def _count_sources(self, data_dir: Path) -> int:
        if not data_dir.is_dir():
            return 0
        num_files = sum(1 for _ in data_dir.glob("*/*.parquet"))
        if num_files == 0:
            return 0
        batches, extra = divmod(num_files, self._file_batch_size)
        return batches + int(extra > 0)

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self._data_root / "train")
        if split is DatasetSplit.VAL:
            return self._count_sources(self._data_root / "val")
        return self._count_sources(self._data_root / "test")


if __name__ == "__main__":
    from dronalize.datasets.argoverse2 import DATASET_SPEC
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env

    root = resolve_dataset_root_from_env("argoverse2")
    _ = debug_descriptor(DATASET_SPEC, root)
