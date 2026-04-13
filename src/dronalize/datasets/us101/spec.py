from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dronalize.config.models import (
    DatasetConfig,
    FullMapExtraction,
    MapConfig,
    ScenesConfig,
)
from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.shared.resources import open_single_shared_map_resource
from dronalize.datasets.shared.specs import (
    lane_change_sampling,
    minimum_samples_screening,
    scenes_config,
)
from dronalize.datasets.us101.loader import US101Loader
from dronalize.datasets.us101.maps.builder import US101MapBuilder
from dronalize.processing.loading.loader import BlockSplitSupport
from dronalize.processing.loading.resources import DatasetResources


@contextmanager
def open_us101_resources(
    root: Path, scenes: ScenesConfig, map_config: MapConfig | None
) -> Generator[DatasetResources, None, None]:
    """Build the shared US-101 map once per run."""
    _ = scenes
    with open_single_shared_map_resource(
        map_config=map_config,
        map_path=root,
        build_map=lambda path, config: US101MapBuilder(path).build(
            config.min_distance, config.interp_distance
        ),
    ) as resources:
        yield resources


DATASET_SPEC = DatasetSpec(
    name="us101",
    loader_factory=US101Loader.unified_factory,
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
    native_schema=US101Loader.native_trajectory_schema(),
    resources_factory=open_us101_resources,
    has_map=True,
    time_split_support=BlockSplitSupport(),
)
