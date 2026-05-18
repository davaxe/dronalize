from dronalize.config.models import DatasetConfig, MapConfig, TrajectoryBufferExtraction
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from dronalize.datasets.shared.presets import minimum_samples_screening, scenes_config
from dronalize.datasets.shared.resources import single_shared_map_resource_factory
from dronalize.datasets.vod.loader import VodLoader, VodLoaderOptions
from dronalize.datasets.vod.maps import VodMapBuilder

_open_vod_resources = single_shared_map_resource_factory(
    map_path=lambda root: root / "maps" / "expansion" / "delft.json",
    build_map=lambda path, config: VodMapBuilder.from_json_file(path).build(
        config.min_distance, config.interpolation_distance
    ),
)


DATASET_DESCRIPTOR = DatasetDescriptor(
    name="vod",
    loader_factory=VodLoader.from_loader_request,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=5, future_frames=30, sample_time=0.1, window_step=5),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=TrajectoryBufferExtraction(radius=25)),
    ),
    loader_options_model=VodLoaderOptions,
    native_schema=VodLoader.native_trajectory_schema(),
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST),
    resources_factory=_open_vod_resources,
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True, source=False, time_block=True),
)
