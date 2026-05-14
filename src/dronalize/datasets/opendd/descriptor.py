from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig
from dronalize.datasets.opendd.loader import OpenDDLoader
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

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="opendd",
    loader_factory=OpenDDLoader.from_loader_request,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=60,
            future_frames=150,
            sample_time=1 / 30,
            window_step=75,
            resample=linear_resample(up=1, down=3),
        ),
        screening=minimum_samples_screening(6),
        map=MapConfig(extraction=FullMapExtraction()),
    ),
    native_schema=OpenDDLoader.native_trajectory_schema(),
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True, source=True),
)
