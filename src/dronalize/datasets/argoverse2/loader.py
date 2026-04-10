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
from dronalize.processing.loading.base import (
    BaseSceneLoader,
    LoaderOptions,
    LoaderSplitCapabilities,
)
from dronalize.processing.loading.loader import LoadedSourceData, MapBinding, Source

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.map_graph import MapGraph
    from dronalize.core.scene import Scene, TrajectorySchema
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.maps.resolver import MapResolver
    from dronalize.processing.models import LoaderRequest


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


class Argoverse2LoaderOptions(LoaderOptions):
    """Dataset-owned config for the Argoverse 2 loader."""

    file_batch_size: int = Field(default=100, ge=1)


class Argoverse2Loader(BaseSceneLoader[list[Path], Argoverse2LoaderOptions]):
    """Loader for Argoverse 2 trajectory data stored in Parquet files."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_scene_split=True
    )

    def __init__(
        self,
        *,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        """Initialize the Argoverse 2 loader."""
        super().__init__(data_root=data_root, request=request, resources=resources)

    @classmethod
    @override
    def loader_options_model(cls) -> type[Argoverse2LoaderOptions]:
        return Argoverse2LoaderOptions

    def _sources_from_dir(self, data_dir: Path) -> Iterable[Source[list[Path]]]:
        if not data_dir.is_dir():
            return
        parquet_files = sorted(data_dir.glob("*/*.parquet"))
        for i in range(0, len(parquet_files), self.loader_options.file_batch_size):
            yield Source(
                identifier=i, data=parquet_files[i : i + self.loader_options.file_batch_size]
            )

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[list[Path]]]:
        if split is DatasetSplit.TRAIN:
            return self._sources_from_dir(self.root / "train")
        if split is DatasetSplit.VAL:
            return self._sources_from_dir(self.root / "val")
        return self._sources_from_dir(self.root / "test")

    @override
    def load_source(self, source: Source[list[Path]]) -> Iterable[LoadedSourceData]:
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
            yield LoadedSourceData(
                frame=group.lazy().drop("file_id"),
                map_binding=MapBinding(map_key=file_to_map.get(str(file_id))),
            )

    @override
    def num_sources(self) -> int | None:
        return sum(
            self._count_sources_for_split(split) for split in self.native_splits or _NATIVE_SPLITS
        )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_YAW

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
        batches, extra = divmod(num_files, self.loader_options.file_batch_size)
        return batches + int(extra > 0)

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self.root / "train")
        if split is DatasetSplit.VAL:
            return self._count_sources(self.root / "val")
        return self._count_sources(self.root / "test")
