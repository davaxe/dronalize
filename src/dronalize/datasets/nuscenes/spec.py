from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dronalize.config.models import (
    DatasetConfig,
    MapConfig,
    SceneExtentExtraction,
    ScenesConfig,
)
from dronalize.datasets.nuscenes.loader import NuScenesLoader, NuScenesLoaderOptions
from dronalize.datasets.nuscenes.maps.builder import NuScenesMapBuilder
from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.shared.resources import open_named_shared_map_resources
from dronalize.datasets.shared.specs import (
    minimum_samples_screening,
    resample_config,
    scenes_config,
)
from dronalize.processing.loading.resources import DatasetResources


@contextmanager
def open_nuscenes_resources(
    root: Path, scenes: ScenesConfig, map_config: MapConfig | None
) -> Generator[DatasetResources, None, None]:
    """Build shared nuScenes maps once per run."""
    _ = scenes
    with open_named_shared_map_resources(
        map_config=map_config,
        named_paths=(
            (path.stem, path)
            for path in (root / "nuScenes-map-expansion-v1.3" / "expansion").glob("*.json")
        ),
        build_map=lambda path, config: NuScenesMapBuilder.from_json_file(path).build(
            config.min_distance, config.interp_distance
        ),
    ) as resources:
        yield resources


DATASET_SPEC = DatasetSpec(
    name="nuscenes",
    loader_factory=NuScenesLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=4,
            future_frames=12,
            sample_time=0.5,
            window_step=1,
            resample=resample_config(method="linear", up=5),
        ),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=SceneExtentExtraction()),
    ),
    dataset_options_model=NuScenesLoaderOptions,
    native_schema=NuScenesLoader.native_trajectory_schema(),
    resources_factory=open_nuscenes_resources,
    has_map=True,
)
