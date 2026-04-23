from dronalize.config.models import DatasetConfig, MapConfig, SceneExtentExtraction
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.nuscenes.loader import NuScenesLoader, NuScenesLoaderOptions
from dronalize.datasets.nuscenes.maps.builder import NuScenesMapBuilder
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.resources import named_shared_map_resources_factory
from dronalize.datasets.shared.specs import (
    linear_resample,
    minimum_samples_screening,
    scenes_config,
)

_open_nuscenes_resources = named_shared_map_resources_factory(
    named_paths=lambda root: (
        (path.stem, path)
        for path in (root / "nuScenes-map-expansion-v1.3" / "expansion").glob("*.json")
    ),
    build_map=lambda path, config: NuScenesMapBuilder.from_json_file(path).build(
        config.min_distance, config.interp_distance
    ),
)


DATASET_SPEC = DatasetSpec(
    name="nuscenes",
    loader_factory=NuScenesLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=4,
            future_frames=12,
            sample_time=0.5,
            window_step=1,
            resample=linear_resample(up=5),
        ),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=SceneExtentExtraction()),
    ),
    dataset_options_model=NuScenesLoaderOptions,
    native_schema=NuScenesLoader.native_trajectory_schema(),
    resources_factory=_open_nuscenes_resources,
    has_map=True,
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL),
    split_support=DatasetSplitSupport(scene=True, source=True),
)
