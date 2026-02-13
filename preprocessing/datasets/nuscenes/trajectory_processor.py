from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import polars as pl

# Adjust imports to match your project structure
from preprocessing.common.trajectory_utils import (
    filter_scene_expr,
    resample_tracks,
    yaw_from_vel,
)
from preprocessing.core.categories import AgentCategory
from preprocessing.core.interface import DataProcessor, ProcessorConfig, Resampling
from preprocessing.datasets.lyft.trajectory_processor import sliding_window

if TYPE_CHECKING:
    from collections.abc import Iterable

    from preprocessing.common.trajectory_utils import T_DataFrame


class NuScenesProcessor(DataProcessor[tuple[str, str], str]):
    """Nuscenes trajectory processor.

    Strategy:
    1. Load all tables globally (cached).
    2. Perform one massive join to process all tracks from all scenes.
    3. Partition result into a Dict[scene_token, DataFrame] for O(1) access.
    """

    def __init__(
        self,
        data_directory: Path | str,
        processor_config: ProcessorConfig | None = None,
        *,
        use_parquet_cache: bool = True,
        parquet_dir: Path | str | None = None,
    ) -> None:
        """Initialize the data processor.

        Args:
            data_directory: data directory containing the raw nuScenes trajectories.
                This is the directory that for example include:
                - category.json
                - instance.json
                - sample_annotation.json
                - sample_data.json
                - ego_pose.json
                - scene.json
                - log.json
            processor_config: configuration for the processor.
            use_parquet_cache: Wether to use parquet cache. This will make
                subsequent processing faster.
            parquet_dir: directory to save the parquet cache files.

        """
        super().__init__(
            processor_config or self._default_config(), enforce_schema=True
        )
        self.data_dir = Path(data_directory)
        self._dfs: dict[str, pl.LazyFrame] = {}
        self._use_parquet = use_parquet_cache
        self._parquet_dir = (
            Path(parquet_dir) if parquet_dir is not None else self.data_dir
        )

        # Cache for the processed data: {scene_token: DataFrame}
        self._scene_cache: dict[str, pl.DataFrame] = {}
        self._schemas: dict[str, pl.Schema | None] = _SCHEMAS

        self._load_tables()
        self._precompute_global_data()

    def _load_tables(self) -> None:
        """Load all required tables using the generic loader."""
        for name, schema in self._schemas.items():
            self._dfs[name] = load_cached_table(
                name=name,
                base_dir=self.data_dir,
                schema=schema,
                use_parquet=self._use_parquet,
                parquet_dir=self._parquet_dir,
            )

    def _precompute_global_data(self) -> None:
        # 1. Build the global timeline (Sample + Scene + Log)
        timeline_lf = build_scene_timeline(
            self._dfs["sample"], self._dfs["scene"], self._dfs["log"]
        )

        # 2. Process Ego
        ego_lf = extract_ego_tracks(
            timeline_lf, self._dfs["sample_data"], self._dfs["ego_pose"]
        )

        # 3. Process Agents (with explicit mappings passed in)
        agents_lf = extract_agent_tracks(
            timeline_lf,
            self._dfs["sample_annotation"],
            self._dfs["instance"],
            self._dfs["category"],
            self._dfs["attribute"],
            category_mapping=_FULL_CATEGORY_MAPPING,
            status_mapping=_STATUS_MAPPING,
            default_agent_category=AgentCategory.UNKNOWN,
            default_status="unknown",
        )

        # 4. Align columns and merge
        # Ensure agents have the exact same column order as ego
        agents_lf = agents_lf.select(ego_lf.columns)
        combined_df = (
            pl
            .concat([ego_lf, agents_lf])
            .sort(["scene_token", "frame", "id"])
            .collect(engine="streaming")
        )

        self._scene_cache = cast(
            "dict[str, pl.DataFrame]",
            combined_df.partition_by("scene_token", as_dict=True),
        )

    @override
    def sources(self) -> Iterable[tuple[tuple[str, str], str]]:
        for token, df in self._scene_cache.items():
            # More efficient: peek at the first row's scene_name directly
            scene_name = df.item(0, "scene_name")
            map_name = df.item(0, "map")
            yield (scene_name, map_name), token

    @override
    def load_raw(self, source: str) -> Iterable[pl.LazyFrame]:
        resampling = self.processor_config.resampling or Resampling(1, 1)
        scenes = (
            self
            ._scene_cache[source]
            .drop([
                "scene_token",
                "scene_name",
                "map",
            ])
            .lazy()
        )
        scenes = scenes.filter(
            pl.col("status") != "parked",
            ~pl.col("full_category").str.contains("object"),
        ).drop(["status", "full_category", "full_status"])

        group_by: list[str] = []
        if self.processor_config.window_params is not None:
            scenes = sliding_window(
                scenes,
                window_size=self.sequence_length,
                step_size=self.processor_config.window_params.step_size,
                sliding_col="frame",
                is_sorted=True,
                return_iterable=False,
            )
            group_by.append("window_index")

        source_filtered = scenes.filter(
            filter_scene_expr(
                self.processor_config,
                group_by=group_by[-1] if len(group_by) > 0 else None,
                category_column="agent_category",
            )
        )
        group_by.append("id")
        source_filtered = source_filtered.filter(pl.len().over(group_by) >= 2)

        processed_source = resample_tracks(
            source_filtered,
            resampling.up,
            resampling.down,
            group_by=group_by,
            add_derivative=True,
            add_second_derivative=True,
            method=resampling.method,
            dt=self.processor_config.sample_time,
            derivative_rename=self.derivative_names(),
            forward_fill=["agent_category"],
        )

        if self.processor_config.window_params is None:
            yield processed_source.lazy()
            return

        for _, group in processed_source.collect().group_by("window_index"):
            yield group.lazy().drop("window_index")

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return yaw_from_vel(df)

    def _default_config(self) -> ProcessorConfig:
        return (
            ProcessorConfig(4, 12, 0.5)
            .resampling_parameters(up=5, down=1)
            .window_parameters(step_size=1)
        )


def load_cached_table(
    name: str,
    base_dir: Path,
    schema: pl.Schema | None = None,
    *,
    use_parquet: bool = True,
    parquet_dir: Path | None = None,
) -> pl.LazyFrame:
    """Load a table from Parquet if available/enabled, otherwise falls back to JSON.

    Args:
        name: name of the JSON-file to read without extension (e.g., "scene")
        base_dir: base directory where the JSON file is located
        schema: schema to use for reading the JSON file. Defaults to None.
        use_parquet: whether to use parquet files if available. Defaults to True.
        parquet_dir: directory where parquet files are stored. If None, defaults to base_dir.

    Returns:
        LazyFrame of the loaded table

    """
    p_dir = parquet_dir if parquet_dir is not None else base_dir
    json_path = base_dir / f"{name}.json"
    parquet_path = p_dir / f"{name}.parquet"

    if use_parquet and parquet_path.exists():
        return pl.scan_parquet(parquet_path)

    # Fallback to JSON
    lf = pl.read_json(json_path, schema=schema)
    lf = lf.lazy()

    if use_parquet:
        # Create directory if it doesn't exist before sinking
        p_dir.mkdir(parents=True, exist_ok=True)
        lf.sink_parquet(parquet_path)
        # Reload from the new parquet to ensure consistency
        return pl.scan_parquet(parquet_path)

    return lf


def build_scene_timeline(
    sample_lf: T_DataFrame,
    scene_lf: T_DataFrame,
    log_lf: T_DataFrame,
) -> T_DataFrame:
    """Build scene timeline.

    Args:
        sample_lf: sample data dataframe.
        scene_lf: scene dataframe.
        log_lf: log dataframe.

    Returns:
        Timeline dataframe for a scene.

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

    Args:
        timeline_lf: timeline dataframe for a scene returned by `build_scene_timeline`
        sample_data_lf: sample data dataframe.
        ego_pose_lf: ego pose dataframe.
        ego_id: ego id. Defaults to 0.
        ego_category: ego category. Defaults to AgentCategory.CAR.

    Returns:
        Track of the ego vehicle.

    """
    return (
        timeline_lf
        .join(
            sample_data_lf.select(["sample_token", "ego_pose_token"]),
            on="sample_token",
        )
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

    Args:
        timeline_lf: return from `build_scene_timeline`
        annotation_lf: sample annotation dataframe.
        instance_lf: instance dataframe.
        category_lf: category dataframe.
        attribute_lf: attribute dataframe.
        category_mapping: mapping from category names to integer categories.
        status_mapping: mapping from status names to string statuses.
        default_agent_category: default integer category for unmapped categories.
            Defaults to AgentCategory.UNKNOWN.
        default_status: default string status for unmapped statuses. Defaults
            to "unknown".

    Returns:
        A LazyFrame containing agent tracks.

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
        .with_columns(
            pl.col("attribute_tokens").list.first().alias("first_attr_token")
        )
        .join(
            attr_lookup,
            left_on="first_attr_token",
            right_on="attr_token",
            how="left",
        )
        .with_columns(
            pl
            .col("instance_token")
            .rank("dense")
            .over("scene_token")
            .cast(pl.Int32)
            .alias("id")
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

_STATUS_MAPPING = {
    "vehicle.moving": "moving",
    "vehicle.stopped": "stopped",
    "vehicle.parked": "parked",
    "pedestrian.moving": "moving",
    "pedestrian.standing": "stopped",
    "pedestrian.sitting_lying_down": "stopped",
    "cycle.with_rider": "moving",
    "cycle.without_rider": "stopped",
}

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
        "translation": pl.List(pl.Float32),
        "size": pl.List(pl.Float32),
        "rotation": pl.List(pl.Float32),
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
    "category": pl.Schema({
        "token": pl.String,
        "name": pl.String,
        "description": pl.String,
    }),
    "attribute": pl.Schema({
        "token": pl.String,
        "name": pl.String,
        "description": pl.String,
    }),
    "ego_pose": pl.Schema({
        "token": pl.String,
        "timestamp": pl.Int64,
        "translation": pl.List(pl.Float32),
        "rotation": pl.List(pl.Float32),
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

if __name__ == "__main__":
    import time

    # Update this path to your actual data location
    data_dir = Path(
        "/home/west/Developer/behavior-prediction/datasets/nuscenes/v1.0-trainval_meta/v1.0-trainval/"
    )

    # Check if directory exists to avoid FileNotFound errors in example
    if data_dir.exists():
        start_time = time.perf_counter()
        processor = NuScenesProcessor(
            data_directory=data_dir,
            use_parquet_cache=True,
            parquet_dir="temp",
        )
        for _scene in processor.scenes_iter():
            break
    else:
        print(f"Path not found: {data_dir}")
