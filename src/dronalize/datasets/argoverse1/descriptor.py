from dronalize.config.models import DatasetConfig, MapConfig, TrajectoryBufferExtraction
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.argoverse1.loader import Argoverse1Loader, Argoverse1LoaderOptions
from dronalize.datasets.argoverse1.maps import Argoverse1MapBuilder
from dronalize.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from dronalize.datasets.shared.presets import minimum_samples_screening, scenes_config
from dronalize.datasets.shared.resources import named_shared_map_resources_factory

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

_open_argoverse1_resources = named_shared_map_resources_factory(
    named_paths=lambda root: (
        ("MIA", root / "hd_maps" / "map_files" / "pruned_argoverse_MIA_10316_vector_map.xml"),
        ("PIT", root / "hd_maps" / "map_files" / "pruned_argoverse_PIT_10314_vector_map.xml"),
    ),
    build_map=lambda path, config: Argoverse1MapBuilder.from_xml_file(path).build(
        config.min_distance, config.interpolation_distance
    ),
)


DATASET_DESCRIPTOR = DatasetDescriptor(
    name="argoverse1",
    loader_factory=Argoverse1Loader.from_loader_request,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=20, future_frames=30, sample_time=0.1),
        screening=minimum_samples_screening(2, prediction_frame=19),
        map=MapConfig(extraction=TrajectoryBufferExtraction(radius=25)),
        loader_options=Argoverse1LoaderOptions().model_dump(),
    ),
    native_schema=Argoverse1Loader.native_trajectory_schema(),
    supported_native_splits=_NATIVE_SPLITS,
    loader_options_model=Argoverse1LoaderOptions,
    resources_factory=_open_argoverse1_resources,
    split_support=DatasetSplitSupport(scene=True),
    feature_support=DatasetFeatureSupport(map=True),
)
