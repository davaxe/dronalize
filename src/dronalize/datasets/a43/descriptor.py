from dronalize.config.models.dataset import DatasetConfig
from dronalize.config.models.map import FullMapExtraction, MapConfig
from dronalize.datasets.a43.loader import A43Loader, A43LoaderOptions
from dronalize.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from dronalize.datasets.shared.presets import minimum_samples_screening, scenes_config

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="a43",
    loader_factory=A43Loader.from_loader_request,
    default_config=DatasetConfig(
        scenes=scenes_config(
            horizon_frames=70, default_observation_length=20, sample_time=0.1, window_step=25
        ),
        screening=minimum_samples_screening(2, required_frame=19),
        map=MapConfig(extraction=FullMapExtraction()),
    ),
    native_schema=A43Loader.native_trajectory_schema(),
    feature_support=DatasetFeatureSupport(map=True),
    loader_options_model=A43LoaderOptions,
    split_support=DatasetSplitSupport(scene=True, time_block=True),
)
