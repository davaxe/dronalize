from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig, ScenesConfig
from dronalize.datasets.i80.loader import I80Loader
from dronalize.datasets.i80.maps.builder import I80MapBuilder
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.resources import open_single_shared_map_resource
from dronalize.datasets.shared.specs import (
    lane_change_sampling,
    minimum_samples_screening,
    scenes_config,
)
from dronalize.processing.loading.resources import DatasetResources


@contextmanager
def open_i80_resources(
    root: Path, scenes: ScenesConfig, map_config: MapConfig | None
) -> Generator[DatasetResources, None, None]:
    """Build the shared I-80 map once per run."""
    _ = scenes
    if map_config is None:
        yield DatasetResources()
        return
    with open_single_shared_map_resource(
        map_config=map_config,
        map_path=root,
        build_map=lambda path, config: I80MapBuilder(path).build(
            config.min_distance, config.interp_distance
        ),
    ) as resources:
        yield resources


DATASET_SPEC = DatasetSpec(
    name="i80",
    loader_factory=I80Loader.unified_runtime_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=20,
            future_frames=50,
            sample_time=0.1,
            window_step=25,
            lane_change=lane_change_sampling(required_lane_changes=3, negative_keep_every=3),
        ),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=FullMapExtraction()),
    ),
    native_schema=I80Loader.native_trajectory_schema(),
    resources_factory=open_i80_resources,
    has_map=True,
    split_support=DatasetSplitSupport(scene=True, time_block=True),
)
