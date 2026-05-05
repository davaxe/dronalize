import polars as pl

from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig
from dronalize.datasets.ngsim.loader import NGSimLoader
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.highway_builder import HighwayLaneMapBuilder, LaneDescription
from dronalize.datasets.shared.resources import ResourcesFactory, single_shared_map_resource_factory
from dronalize.datasets.shared.specs import (
    lane_change_sampling,
    minimum_samples_screening,
    scenes_config,
)

_SELECT_EXPR = (
    pl.col("Vehicle_ID").alias("id"),
    pl.col("Local_X").alias("x").mul(0.3048),
    pl.col("Local_Y").alias("y").mul(0.3048),
    pl.col("Lane_ID").alias("lane_id"),
)

_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(
        history_frames=20,
        future_frames=50,
        sample_time=0.1,
        window_step=25,
        lane_change=lane_change_sampling(required_lane_changes=3, negative_keep_every=3),
    ),
    screening=minimum_samples_screening(2),
    map=MapConfig(extraction=FullMapExtraction()),
)


def ngsim_resources(
    lane_description: LaneDescription | None = None,
    bin_size: float = 10,
    smoothing: float | None = 3.0,
    *,
    include_outer_borders: bool = True,
) -> ResourcesFactory:
    """Create a resources factory for the NGSIM dataset."""
    return single_shared_map_resource_factory(
        map_path=lambda root: root,
        build_map=lambda path, config: HighwayLaneMapBuilder.from_csv(
            *path.rglob("trajectories*.csv"),
            select_expr=_SELECT_EXPR,
            bin_size=bin_size,
            lane_description=lane_description,
            include_outer_borders=include_outer_borders,
            smoothing=smoothing,
        ).build(config.min_distance, config.interp_distance),
    )


DATASET_SPECS = {
    "i80": DatasetSpec(
        name="i80",
        loader_factory=NGSimLoader.unified_factory,
        default_config=_DEFAULT_CONFIG,
        native_schema=NGSimLoader.native_trajectory_schema(),
        resources_factory=ngsim_resources(
            LaneDescription(ids=list(range(1, 8)), direction=[True] * 7)
        ),
        has_map=True,
        split_support=DatasetSplitSupport(scene=True, time_block=True),
    ),
    "us101": DatasetSpec(
        name="us101",
        loader_factory=NGSimLoader.unified_factory,
        default_config=_DEFAULT_CONFIG,
        native_schema=NGSimLoader.native_trajectory_schema(),
        resources_factory=ngsim_resources(
            LaneDescription(ids=list(range(1, 9)), direction=[True] * 8)
        ),
        has_map=True,
        split_support=DatasetSplitSupport(scene=True, time_block=True),
    ),
}
