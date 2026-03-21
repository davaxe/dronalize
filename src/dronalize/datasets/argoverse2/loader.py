from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.argoverse2.map.builder import Argoverse2MapBuilder
from dronalize.datasets.common import utils
from dronalize.loading import BaseSceneLoader, IngestOutput, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline
from dronalize.scene import POSITIONS_VELOCITY_YAW_V1

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.maps.graph import MapGraph
    from dronalize.maps.resolver import MapResolver
    from dronalize.scene import Scene, SceneSchema


class Argoverse2Loader(BaseSceneLoader[list[Path]]):
    """Loader for Argoverse 2 trajectory data stored in Parquet files."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        *,
        file_batch_size: int | None = 100,
    ) -> None:
        """Initialize the dataset loader.

        Parameters
        ----------
        data_root : Path or str
            Path to the root directory of the Argoverse 2 dataset.
            This directory should contain `train/`, `val/`, and `test/`
            subdirectories with Parquet files.
        file_batch_size : int, optional
            Number of files to process in each batch.
        loader_config : LoaderConfig, optional
            Configuration for the loader.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. Can contain one or more predefined splits,
            or `None` to process all sources.

        """
        super().__init__(loader_config=loader_config, map_config=map_config, splits=splits)
        self._data_root: Path = Path(data_root)
        self._file_batch_size: int | None = file_batch_size

    @classmethod
    @override
    def predefined_splits(cls) -> tuple[DatasetSplit, ...]:
        return (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

    def _sources_from_dir(self, data_dir: Path) -> Iterable[Source[list[Path]]]:
        if not data_dir.is_dir():
            return
        parquet_files = sorted(data_dir.glob("*/*.parquet"))
        batch_size: int = self._file_batch_size or len(parquet_files) or 1
        for i in range(0, len(parquet_files), batch_size):
            batch_files = parquet_files[i : i + batch_size]
            yield Source(identifier=i, inner=batch_files)

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[list[Path]]]:
        if split is DatasetSplit.TRAIN:
            return self._sources_from_dir(self._data_root / "train")
        if split is DatasetSplit.VAL:
            return self._sources_from_dir(self._data_root / "val")
        return self._sources_from_dir(self._data_root / "test")

    @override
    def ingest(self, source: Source[list[Path]]) -> Iterable[IngestOutput]:
        # Build a mapping from parquet file path to its co-located JSON map path.
        # Each parquet lives in a subdirectory like <scenario_id>/<scenario_id>.parquet
        # alongside a <scenario_id>.json map file.
        file_to_map: dict[str, str] = {}
        for pq in source.inner:
            json_candidates = list(pq.parent.glob("*.json"))
            if json_candidates:
                file_to_map[str(pq)] = str(json_candidates[0])

        batch_lf = pl.scan_parquet(source.inner, include_file_paths="file_id").select(
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
            yield group.lazy().drop("file_id"), map_path

    @override
    def num_sources(self) -> int | None:
        splits = self.splits if self.splits is not None else self.predefined_splits()
        return sum(self._count_sources_for_split(split) for split in splits)

    @override
    def pipeline(self) -> Pipeline:
        return Pipeline().compose(trajectory_pipeline(self.loader_config))

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_VELOCITY_YAW_V1

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(input_len=50, output_len=60, sample_time=0.1).with_filtering(
            require_frames=[49],
            exclude_agent_categories=[
                AgentCategory.STATIC_OBJECT,
                AgentCategory.UNKNOWN,
                AgentCategory.UNIMPORTANT,
            ],
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.no_extraction()

    @override
    def map_resolver(self) -> MapResolver:
        def _resolver(scene: Scene) -> MapGraph | None:
            if scene.map_key is None:
                return None

            return utils.extract_based_on_scene(
                self._get_map(
                    scene.map_key,
                    self.map_config.min_distance,
                    self.map_config.interp_distance,
                ),
                scene,
                self.map_config.extraction,
            )

        return _resolver

    @staticmethod
    @functools.lru_cache(maxsize=10)
    def _get_map(
        key: str,
        min_distance: float | None,
        interp_distance: float | None,
    ) -> MapGraph:
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
            mapping,
            default=AgentCategory.UNKNOWN,
            return_dtype=pl.Int32,
        )

    def _count_sources(self, data_dir: Path) -> int:
        if not data_dir.is_dir():
            return 0
        num_files = sum(1 for _ in data_dir.glob("*/*.parquet"))
        if num_files == 0:
            return 0
        batch_size = self._file_batch_size or num_files
        batches, extra = divmod(num_files, batch_size)
        return batches + int(extra > 0)

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self._data_root / "train")
        if split is DatasetSplit.VAL:
            return self._count_sources(self._data_root / "val")
        return self._count_sources(self._data_root / "test")
