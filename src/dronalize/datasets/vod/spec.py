from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dronalize.config.models import (
    DatasetConfig,
    MapConfig,
    SceneExtentExtraction,
    ScenesConfig,
)
from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.shared.resources import open_single_shared_map_resource
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config
from dronalize.datasets.vod.loader import VodLoader
from dronalize.datasets.vod.maps.builder import VODMapBuilder
from dronalize.processing.loading.resources import DatasetResources


@contextmanager
def open_vod_resources(
    root: Path, scenes: ScenesConfig, map_config: MapConfig | None
) -> Generator[DatasetResources, None, None]:
    """Build the shared VOD map once per run."""
    _ = scenes
    if map_config is None:
        yield DatasetResources()
        return
    with open_single_shared_map_resource(
        map_config=map_config,
        map_path=root / "maps" / "expansion" / "delft.json",
        build_map=lambda path, config: VODMapBuilder.from_json_file(path).build(
            config.min_distance, config.interp_distance
        ),
    ) as resources:
        yield resources


DATASET_SPEC = DatasetSpec(
    name="vod",
    loader_factory=VodLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=5, future_frames=30, sample_time=0.1, window_step=5),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=SceneExtentExtraction()),
    ),
    native_schema=VodLoader.native_trajectory_schema(),
    resources_factory=open_vod_resources,
    has_map=True,
)
