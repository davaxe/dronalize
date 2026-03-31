from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_ONLY_V1
from dronalize.processing.filters import Filter
from dronalize.processing.filters.agent import RequireFrames
from dronalize.processing.ingest.base import BaseSceneLoader, LoaderSplitCapabilities
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.ingest.loader import IngestedData, Source
from dronalize.processing.maps.resolver import no_map, shared_map
from dronalize.processing.pipeline.functional.resample import ResampleSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.scene import SceneSchema
    from dronalize.processing.ingest.splits import SplitConfig
    from dronalize.processing.maps.config import MapConfig
    from dronalize.processing.maps.resolver import MapResolver


class NuScenesLoader(BaseSceneLoader[tuple[int, str]]):
    """Loader for nuScenes trajectories.

    The loader preprocesses the global nuScenes tables once, joins them into a
    scene-level trajectory table, and then caches one DataFrame per scene for
    efficient access during ingestion.
    """

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
        """Initialize the nuScenes loader.

        Parameters
        ----------
        data_root : Path or str
            Root directory of the extracted nuScenes dataset.
        loader_config : LoaderConfig, optional
            Loader configuration override.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Optional selection of predefined dataset splits. This loader does
            not expose predefined splits, so `None` processes all sources.
        """
        super().__init__(
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )
        self._data_root: Path = Path(data_root)
        self._data_dirs: list[Path] = self._find_data_dir()
        self._dfs: list[dict[str, pl.LazyFrame]] = []

        # Cache for the processed data: {scene_token: DataFrame}
        self._scene_cache: list[dict[str, pl.DataFrame]] = []
        self._schemas: dict[str, pl.Schema | None] = _SCHEMAS

        self._load_tables()
        self._precompute_global_data()

    def _find_data_dir(self) -> list[Path]:
        required_files = set(_SCHEMAS.keys())
        paths: list[Path] = []
        max_level = 3
        current_root = self._data_root
        for level in range(max_level + 1):
            pattern = "*/" * level + "*"

            for path in current_root.glob(pattern):
                if path.is_dir():
                    files = {p.stem for p in path.glob("*.json")}
                    if required_files.issubset(files):
                        paths.append(path)

        return paths

    @override
    def discover_sources(self) -> Iterable[Source[tuple[int, str]]]:
        for i, dfs in enumerate(self._scene_cache):
            for token, df in dfs.items():
                scene_name: str = df.item(0, "scene_name")
                map_name: str = df.item(0, "map")
                yield Source(identifier=scene_name, data=(i, token), map_key=map_name)

    @override
    def num_sources(self) -> int | None:
        return sum(len(dfs) for dfs in self._scene_cache)

    @override
    def ingest(self, source: Source[tuple[int, str]]) -> Iterable[IngestedData]:
        index, token = source.data
        scenes = self._scene_cache[index][token].drop(["scene_token", "scene_name", "map"]).lazy()
        yield IngestedData(
            scenes.filter(
                ~pl.col("status").is_in(self._status_to_filter),
                *[
                    ~pl.col("full_category").str.contains(category)
                    for category in self._full_category_contains
                ],
            ).drop(["status", "full_category", "full_status"])
        )

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_ONLY_V1

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=4, output_len=12, sample_time=0.5)
            .with_resampling(ResampleSpec(up=5, down=1))
            .with_window(step=1)
            .with_filter(Filter.define(agent_rules=[RequireFrames.define(frames=[3])]))
        )

    @override
    def map_resolver(self) -> MapResolver:
        if self._shared_memory_name is None:
            return no_map()
        return shared_map(self._shared_memory_name)

    def _load_tables(self) -> None:
        """Load all required tables using the generic loader."""
        for data_dir in self._data_dirs:
            data_dict: dict[str, pl.LazyFrame] = {}
            for name, schema in self._schemas.items():
                data_dict[name] = load_cached_table(name=name, base_dir=data_dir, schema=schema)
            self._dfs.append(data_dict)

    def _precompute_global_data(self) -> None:
        # 1. Build the global timeline (Sample + Scene + Log)
        for dfs in self._dfs:
            timeline_lf = build_scene_timeline(dfs["sample"], dfs["scene"], dfs["log"])

            # 2. Process Ego
            ego_lf = extract_ego_tracks(timeline_lf, dfs["sample_data"], dfs["ego_pose"])

            # 3. Process Agents (with explicit mappings passed in)
            agents_lf = extract_agent_tracks(
                timeline_lf,
                dfs["sample_annotation"],
                dfs["instance"],
                dfs["category"],
                dfs["attribute"],
                category_mapping=_FULL_CATEGORY_MAPPING,
                status_mapping=_STATUS_MAPPING,
                default_agent_category=AgentCategory.UNKNOWN,
                default_status="unknown",
            )

            # 4. Align columns and merge
            # Ensure agents have the exact same column order as ego
            agents_lf = agents_lf.select(ego_lf.collect_schema().names())
            combined_df = (
                pl
                .concat([ego_lf, agents_lf])
                .sort(["scene_token", "frame", "id"])
                .collect(engine="streaming")
            )

            self._scene_cache.append(
                cast(
                    "dict[str, pl.DataFrame]", combined_df.partition_by("scene_token", as_dict=True)
                )
            )

        self._status_to_filter: list[str] = ["parked", "undefined"]
        self._full_category_contains: list[str] = ["object"]


def load_cached_table(name: str, base_dir: Path, schema: pl.Schema | None = None) -> pl.LazyFrame:
    """Load a table from Parquet if available/enabled, otherwise falls back to JSON.

    Parameters
    ----------
    name : str
        Name of the JSON file to read without extension (e.g., "scene").
    base_dir : Path
        Base directory where the JSON file is located.
    schema : pl.Schema, optional
        Schema to use for reading the JSON file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame of the loaded table.

    """
    json_path = base_dir / f"{name}.json"

    # Fallback to JSON
    lf = pl.read_json(json_path, schema=schema)
    return lf.lazy()


def build_scene_timeline(
    sample_lf: pl.LazyFrame, scene_lf: pl.LazyFrame, log_lf: pl.LazyFrame
) -> pl.LazyFrame:
    """Build a frame-indexed scene timeline table.

    Parameters
    ----------
    sample_lf : pl.LazyFrame
        Sample data LazyFrame.
    scene_lf : pl.LazyFrame
        Scene LazyFrame.
    log_lf : pl.LazyFrame
        Log LazyFrame.

    Returns
    -------
    pl.LazyFrame
        Timeline table containing scene, sample, and log metadata.
    """
    return (
        sample_lf
        .join(scene_lf, left_on="scene_token", right_on="token")
        .join(log_lf, left_on="log_token", right_on="token")
        .select([
            pl.col("token").alias("sample_token"),
            pl.col("scene_token"),
            pl.col("name").alias("scene_name"),
            pl.col("location").alias("map"),
            pl.col("timestamp"),
        ])
        .with_columns(
            pl
            .col("timestamp")
            .rank("dense")
            .over("scene_token")
            .sub(1)
            .alias("frame")
            .cast(pl.Int64)
        )
    )


def extract_ego_tracks(
    timeline_lf: pl.LazyFrame,
    sample_data_lf: pl.LazyFrame,
    ego_pose_lf: pl.LazyFrame,
    ego_id: int = 0,
    ego_category: int = AgentCategory.CAR,
) -> pl.LazyFrame:
    """Extract ego track from relevant dataframes.

    Parameters
    ----------
    timeline_lf : pl.LazyFrame
        Timeline LazyFrame for a scene, as returned by `build_scene_timeline`.
    sample_data_lf : pl.LazyFrame
        Sample data LazyFrame.
    ego_pose_lf : pl.LazyFrame
        Ego pose LazyFrame.
    ego_id : int, optional
        Ego agent ID. Defaults to 0.
    ego_category : int, optional
        Ego agent category. Defaults to `AgentCategory.CAR`.

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing the track of the ego vehicle.

    """
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


def extract_agent_tracks(
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
    """Extract agent tracks from relevant tables.

    Parameters
    ----------
    timeline_lf : pl.LazyFrame
        Timeline LazyFrame, as returned by `build_scene_timeline`.
    annotation_lf : pl.LazyFrame
        Sample annotation LazyFrame.
    instance_lf : pl.LazyFrame
        Instance LazyFrame.
    category_lf : pl.LazyFrame
        Category LazyFrame.
    attribute_lf : pl.LazyFrame
        Attribute LazyFrame.
    category_mapping : dict[str, int]
        Mapping from category names to integer categories.
    status_mapping : dict[str, str]
        Mapping from status names to string statuses.
    default_agent_category : int, optional
        Default integer category for unmapped categories.
        Defaults to `AgentCategory.UNKNOWN`.
    default_status : str, optional
        Default string status for unmapped statuses. Defaults to "unknown".

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing agent tracks.

    """
    # 1. Prepare small lookup tables
    attr_lookup = attribute_lf.select([
        pl.col("token").alias("attr_token"),
        pl.col("name").alias("attr_name"),
    ])
    cat_lookup = category_lf.select([
        pl.col("token").alias("cat_token"),
        pl.col("name").alias("cat_name"),
    ])

    # 2. Main join
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

    # 3. Apply Mappings
    # We use replace_strict with a default=None to catch unmapped values as nulls
    # before filling them with the integer default.
    return raw_agents.with_columns([
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
    ])


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
    # Vehicles
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
    # Humans
    "human.pedestrian.adult": AgentCategory.PEDESTRIAN,
    "human.pedestrian.child": AgentCategory.PEDESTRIAN,
    "human.pedestrian.construction_worker": AgentCategory.PEDESTRIAN,
    "human.pedestrian.police_officer": AgentCategory.PEDESTRIAN,
    "human.pedestrian.stroller": AgentCategory.PEDESTRIAN,
    "human.pedestrian.wheelchair": AgentCategory.PEDESTRIAN,
    "human.pedestrian": AgentCategory.PEDESTRIAN,  # Catch-all
    # Static / Other
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


if __name__ == "__main__":
    from dronalize.datasets.nuscenes import DESCRIPTOR
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env

    root = resolve_dataset_root_from_env("nuscenes")
    _ = debug_descriptor(DESCRIPTOR, root)
