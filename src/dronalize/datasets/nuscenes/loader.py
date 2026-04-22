from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import Field
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_ONLY
from dronalize.datasets.nuscenes.splits import get_split
from dronalize.datasets.shared import utils
from dronalize.processing.loading.base import ALL_SOURCES, BaseSceneLoader, SourceSelection
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.processing.loading.options import DatasetOptionsModel
from dronalize.processing.maps.resolver import no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.maps.resolver import MapResolver
    from dronalize.processing.models import LoaderRequest

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL)


class NuScenesLoaderOptions(DatasetOptionsModel):
    drop_status: list[str] = Field(default_factory=list)
    drop_full_category_regex: list[str] = Field(default_factory=lambda: ["object"])


class NuScenesLoader(BaseSceneLoader[str, NuScenesLoaderOptions]):
    """Loader for nuScenes trajectories."""

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        super().__init__(data_root=data_root, request=request, resources=resources)
        self._schemas: dict[str, pl.Schema | None] = _SCHEMAS
        # Lightweight manifest used by discover_sources/num_sources.
        self._sources: dict[DatasetSplit, list[Source[str]]] | None = None
        # Heavy process-local caches, initialized lazily.
        self._scene_cache: dict[str, pl.DataFrame] = {}
        self._data_dir: Path = self.root / "v1.0-trainval_meta" / "v1.0-trainval"

    @classmethod
    @override
    def unified_factory(
        cls,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> NuScenesLoader:
        return cls(data_root, request, resources)

    def _build_sources_manifest(self) -> dict[DatasetSplit, list[Source[str]]]:
        sources: dict[DatasetSplit, list[Source[str]]] = {split: [] for split in _NATIVE_SPLITS}

        with (self._data_dir / "log.json").open("r", encoding="utf-8") as f:
            log_records: list[dict[str, Any]] = json.load(f)
        with (self._data_dir / "scene.json").open("r", encoding="utf-8") as f:
            scene_records: list[dict[str, Any]] = json.load(f)

        log_to_map = {
            str(row["token"]): str(row["location"]) if row.get("location") else None
            for row in log_records
            if "token" in row
        }

        for row in scene_records:
            if "token" not in row or "log_token" not in row or "name" not in row:
                continue
            name = str(row["name"])
            sources[get_split(name)].append(
                Source(
                    identifier=name,
                    data=str(row["token"]),
                    map_key=log_to_map.get(str(row["log_token"])),
                )
            )
        return sources

    @override
    def iter_sources_for(self, selection: SourceSelection = ALL_SOURCES) -> Iterable[Source[str]]:
        split = selection.native_split
        if self._sources is None:
            self._sources = self._build_sources_manifest()
        if split is None:
            for native_split in _NATIVE_SPLITS:
                yield from self._sources.get(native_split, [])
            return
        yield from self._sources.get(split, [])

    @override
    def count_sources_for(self, selection: SourceSelection = ALL_SOURCES) -> int | None:
        if self._sources is None:
            self._sources = self._build_sources_manifest()
        split = selection.native_split
        if split is None:
            return sum(len(self._sources.get(native_split, [])) for native_split in _NATIVE_SPLITS)
        return len(self._sources.get(split, []))

    @override
    def load_source(self, source: Source[str]) -> Iterable[LoadedSourceData]:
        self._ensure_dataset_loaded()
        scenes = self._scene_cache[source.data].drop(["scene_token", "scene_name", "map"]).lazy()
        filters = [~pl.col("status").is_in(self.loader_options.drop_status)]
        filters.extend(
            ~pl.col("full_category").str.contains(regex)
            for regex in self.loader_options.drop_full_category_regex
        )
        yield LoadedSourceData(
            scenes.filter(*filters).drop(["status", "full_category", "full_status"])
        )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY

    @override
    def map_resolver(self) -> MapResolver:
        if not self.resources.shared_maps or self.map_config is None:
            return no_map()
        extraction = self.map_config.extraction
        return shared_map(self.resources.shared_maps, utils.extract_fn(extraction))

    def _ensure_dataset_loaded(self) -> None:
        if self._scene_cache:
            return
        dfs = {
            name: _load_cached_table(name, self._data_dir, schema)
            for name, schema in self._schemas.items()
        }
        timeline_lf = _build_scene_timeline(dfs["sample"], dfs["scene"], dfs["log"])
        ego_lf = _extract_ego_tracks(timeline_lf, dfs["sample_data"], dfs["ego_pose"])
        agents_lf = _extract_agent_tracks(
            timeline_lf,
            dfs["sample_annotation"],
            dfs["instance"],
            dfs["category"],
            dfs["attribute"],
            category_mapping=_FULL_CATEGORY_MAPPING,
            status_mapping=_STATUS_MAPPING,
        ).select(ego_lf.collect_schema().names())
        combined_df = (
            pl
            .concat([ego_lf, agents_lf])
            .sort(["scene_token", "frame", "id"])
            .collect(engine="streaming")
        )
        raw_partitions = combined_df.partition_by("scene_token", as_dict=True)
        self._scene_cache = {key[0]: df for key, df in raw_partitions.items()}


def _load_cached_table(name: str, base_dir: Path, schema: pl.Schema | None = None) -> pl.LazyFrame:
    return pl.read_json(base_dir / f"{name}.json", schema=schema).lazy()


def _build_scene_timeline(
    sample_lf: pl.LazyFrame, scene_lf: pl.LazyFrame, log_lf: pl.LazyFrame
) -> pl.LazyFrame:
    return (
        sample_lf
        .join(scene_lf, left_on="scene_token", right_on="token")
        .join(log_lf, left_on="log_token", right_on="token")
        .select(
            pl.col("token").alias("sample_token"),
            "scene_token",
            pl.col("name").alias("scene_name"),
            pl.col("location").alias("map"),
            "timestamp",
        )
        .with_columns(
            pl
            .col("timestamp")
            .rank("dense")
            .over("scene_token")
            .sub(1)
            .cast(pl.Int64)
            .alias("frame")
        )
    )


def _extract_ego_tracks(
    timeline_lf: pl.LazyFrame,
    sample_data_lf: pl.LazyFrame,
    ego_pose_lf: pl.LazyFrame,
    ego_id: int = 0,
    ego_category: int = AgentCategory.CAR,
) -> pl.LazyFrame:
    return (
        timeline_lf
        .join(sample_data_lf.select(["sample_token", "ego_pose_token"]), on="sample_token")
        .group_by("sample_token", maintain_order=False)
        .first()
        .join(ego_pose_lf, left_on="ego_pose_token", right_on="token")
        .select(
            *("scene_token", "scene_name", "map", "frame"),
            pl.lit(ego_id, dtype=pl.Int32).alias("id"),
            pl.col("translation").list.get(0).alias("x"),
            pl.col("translation").list.get(1).alias("y"),
            pl.lit(ego_category, dtype=pl.Int32).alias("agent_category"),
            pl.lit("vehicle.ego.car").alias("full_category"),
            pl.lit("moving").alias("status"),
            pl.lit("moving").alias("full_status"),
        )
    )


def _extract_agent_tracks(
    timeline_lf: pl.LazyFrame,
    annotation_lf: pl.LazyFrame,
    instance_lf: pl.LazyFrame,
    category_lf: pl.LazyFrame,
    attribute_lf: pl.LazyFrame,
    category_mapping: dict[str, int],
    status_mapping: dict[str, str],
    default_agent_category: int = AgentCategory.UNKNOWN,
    default_status: str = "unknown",
) -> pl.LazyFrame:
    attr_lookup = attribute_lf.select(
        pl.col("token").alias("attr_token"), pl.col("name").alias("attr_name")
    )
    cat_lookup = category_lf.select(
        pl.col("token").alias("cat_token"), pl.col("name").alias("cat_name")
    )
    raw_agents = (
        annotation_lf
        .join(timeline_lf, on="sample_token")
        .join(instance_lf, left_on="instance_token", right_on="token")
        .join(cat_lookup, left_on="category_token", right_on="cat_token")
        .with_columns(pl.col("attribute_tokens").list.first().alias("first_attr_token"))
        .join(attr_lookup, left_on="first_attr_token", right_on="attr_token", how="left")
        .with_columns(
            pl.col("instance_token").rank("dense").over("scene_token").cast(pl.Int32).alias("id")
        )
        .select(
            *("scene_token", "scene_name", "map", "frame", "id"),
            pl.col("translation").list.get(0).alias("x"),
            pl.col("translation").list.get(1).alias("y"),
            pl.col("cat_name").alias("full_category"),
            pl.col("attr_name").fill_null("unknown").alias("full_status"),
        )
    )

    return raw_agents.with_columns(
        pl
        .col("full_category")
        .replace_strict(category_mapping, default=None)
        .fill_null(default_agent_category)
        .cast(pl.Int32)
        .alias("agent_category"),
        pl
        .col("full_status")
        .replace_strict(status_mapping, default=default_status)
        .alias("status"),
    )


_SCHEMAS: dict[str, pl.Schema | None] = {
    "scene": pl.Schema({
        "token": pl.String,
        "log_token": pl.String,
        "nbr_samples": pl.Int32,
        "first_sample_token": pl.String,
        "last_sample_token": pl.String,
        "name": pl.String,
        "description": pl.String,
    }),
    "sample_annotation": pl.Schema({
        "token": pl.String,
        "sample_token": pl.String,
        "instance_token": pl.String,
        "visibility_token": pl.String,
        "attribute_tokens": pl.List(pl.String),
        "translation": pl.List(pl.Float64),
        "size": pl.List(pl.Float64),
        "rotation": pl.List(pl.Float64),
        "prev": pl.String,
        "next": pl.String,
        "num_lidar_pts": pl.Int32,
        "num_radar_pts": pl.Int32,
    }),
    "instance": pl.Schema({
        "token": pl.String,
        "category_token": pl.String,
        "nbr_annotations": pl.Int32,
        "first_annotation_token": pl.String,
        "last_annotation_token": pl.String,
    }),
    "category": pl.Schema({"token": pl.String, "name": pl.String, "description": pl.String}),
    "attribute": pl.Schema({"token": pl.String, "name": pl.String, "description": pl.String}),
    "ego_pose": pl.Schema({
        "token": pl.String,
        "timestamp": pl.Int64,
        "translation": pl.List(pl.Float64),
        "rotation": pl.List(pl.Float64),
    }),
    "log": pl.Schema({
        "token": pl.String,
        "logfile": pl.String,
        "vehicle": pl.String,
        "date_captured": pl.String,
        "location": pl.String,
    }),
    "sample": pl.Schema({
        "token": pl.String,
        "prev": pl.String,
        "next": pl.String,
        "scene_token": pl.String,
        "timestamp": pl.Int64,
    }),
    "sample_data": pl.Schema({
        "token": pl.String,
        "sample_token": pl.String,
        "ego_pose_token": pl.String,
        "calibrated_sensor_token": pl.String,
        "timestamp": pl.Int64,
        "fileformat": pl.String,
        "filename": pl.String,
        "prev": pl.String,
        "next": pl.String,
        "is_key_frame": pl.Boolean,
        "height": pl.Int32,
        "width": pl.Int32,
    }),
}

_FULL_CATEGORY_MAPPING: dict[str, int] = {
    "vehicle.car": AgentCategory.CAR,
    "vehicle.ego.car": AgentCategory.CAR,
    "vehicle.van": AgentCategory.VAN,
    "vehicle.construction": AgentCategory.VAN,
    "vehicle.bus.bendy": AgentCategory.BUS,
    "vehicle.bus.rigid": AgentCategory.BUS,
    "vehicle.truck": AgentCategory.TRUCK,
    "vehicle.trailer": AgentCategory.TRAILER,
    "vehicle.motorcycle": AgentCategory.MOTORCYCLE,
    "vehicle.bicycle": AgentCategory.BICYCLE,
    "vehicle.emergency.ambulance": AgentCategory.CAR,
    "vehicle.emergency.police": AgentCategory.CAR,
    "human.pedestrian.adult": AgentCategory.PEDESTRIAN,
    "human.pedestrian.child": AgentCategory.PEDESTRIAN,
    "human.pedestrian.construction_worker": AgentCategory.PEDESTRIAN,
    "human.pedestrian.police_officer": AgentCategory.PEDESTRIAN,
    "human.pedestrian.stroller": AgentCategory.PEDESTRIAN,
    "human.pedestrian.wheelchair": AgentCategory.PEDESTRIAN,
    "human.pedestrian": AgentCategory.PEDESTRIAN,
    "static_object.bicycle_rack": AgentCategory.STATIC_OBJECT,
    "movable_object.barrier": AgentCategory.MOVEABLE_OBJECT,
    "movable_object.debris": AgentCategory.MOVEABLE_OBJECT,
    "movable_object.pushable_pullable": AgentCategory.MOVEABLE_OBJECT,
    "movable_object.trafficcone": AgentCategory.MOVEABLE_OBJECT,
    "animal": AgentCategory.ANIMAL,
}

_STATUS_MAPPING: dict[str, str] = {
    "vehicle.moving": "moving",
    "vehicle.stopped": "stopped",
    "vehicle.parked": "parked",
    "pedestrian.moving": "moving",
    "pedestrian.standing": "stopped",
    "pedestrian.sitting_lying_down": "stopped",
    "cycle.with_rider": "moving",
    "cycle.without_rider": "stopped",
}
