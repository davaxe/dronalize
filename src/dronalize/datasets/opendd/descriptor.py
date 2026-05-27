from dronalize.config.models.dataset import DatasetConfig
from dronalize.config.models.map import FullMapExtraction, MapConfig
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
    temporal_support,
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="opendd",
    loader_factory=OpenDDLoader.from_loader_request,
    default_config=DatasetConfig(
        scenes=scenes_config(
            horizon_frames=210,
            default_observation_length=60,
            sample_time=1 / 30,
            window_step=75,
            resample=linear_resample(up=1, down=3),
        ),
        screening=minimum_samples_screening(6, required_frame=59),
        map=MapConfig(extraction=FullMapExtraction()),
    ),
    native_schema=OpenDDLoader.native_trajectory_schema(),
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True, source=True, time_block=True),
    temporal_support=temporal_support(
        source_unit="recording", min_frames=456, max_frames=16916, enabled_by_default=True
    ),
)
