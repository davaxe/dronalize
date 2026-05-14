from dronalize.config.models import DatasetConfig, MapConfig, TrajectoryBufferExtraction
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.nuscenes.loader import NuScenesLoader, NuScenesLoaderOptions
from dronalize.datasets.nuscenes.maps import NuScenesMapBuilder
from dronalize.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from dronalize.datasets.shared.presets import (
    linear_resample,
    minimum_samples_screening,
    scenes_config,
)
from dronalize.datasets.shared.resources import named_shared_map_resources_factory

_open_nuscenes_resources = named_shared_map_resources_factory(
    named_paths=lambda root: (
        (path.stem, path)
        for path in (root / "nuScenes-map-expansion-v1.3" / "expansion").glob("*.json")
    ),
    build_map=lambda path, config: NuScenesMapBuilder.from_json_file(path).build(
        config.min_distance, config.interpolation_distance
    ),
)


DATASET_DESCRIPTOR = DatasetDescriptor(
    name="nuscenes",
    loader_factory=NuScenesLoader.from_loader_request,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=4,
            future_frames=12,
            sample_time=0.5,
            window_step=1,
            resample=linear_resample(up=5),
        ),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=TrajectoryBufferExtraction(radius=25)),
    ),
    loader_options_model=NuScenesLoaderOptions,
    native_schema=NuScenesLoader.native_trajectory_schema(),
    resources_factory=_open_nuscenes_resources,
    feature_support=DatasetFeatureSupport(map=True),
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL),
    split_support=DatasetSplitSupport(scene=True, source=True),
)
