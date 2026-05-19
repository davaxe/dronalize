from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.argoverse2.loader import Argoverse2Loader, Argoverse2LoaderOptions
from dronalize.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from dronalize.datasets.shared.presets import minimum_samples_screening, scenes_config

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="argoverse2",
    loader_factory=Argoverse2Loader.from_loader_request,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=50, future_frames=60, sample_time=0.1),
        screening=minimum_samples_screening(2, prediction_frame=49),
        map=MapConfig(extraction=FullMapExtraction()),
        loader_options=Argoverse2LoaderOptions().model_dump(),
    ),
    native_schema=Argoverse2Loader.native_trajectory_schema(),
    supported_native_splits=_NATIVE_SPLITS,
    loader_options_model=Argoverse2LoaderOptions,
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True),
)
