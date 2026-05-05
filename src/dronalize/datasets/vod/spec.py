from dronalize.config.models import DatasetConfig, MapConfig, TrajectoryBufferExtraction
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.resources import single_shared_map_resource_factory
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config
from dronalize.datasets.vod.loader import VodLoader, VodLoaderOptions
from dronalize.datasets.vod.maps.builder import VODMapBuilder

_open_vod_resources = single_shared_map_resource_factory(
    map_path=lambda root: root / "maps" / "expansion" / "delft.json",
    build_map=lambda path, config: VODMapBuilder.from_json_file(path).build(
        config.min_distance, config.interp_distance
    ),
)


DATASET_SPEC = DatasetSpec(
    name="vod",
    loader_factory=VodLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=5, future_frames=30, sample_time=0.1, window_step=5),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=TrajectoryBufferExtraction(radius=25)),
    ),
    dataset_options_model=VodLoaderOptions,
    native_schema=VodLoader.native_trajectory_schema(),
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST),
    resources_factory=_open_vod_resources,
    has_map=True,
    split_support=DatasetSplitSupport(scene=True, source=True),
)
