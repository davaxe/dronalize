from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig
from dronalize.datasets.ad4che.loader import AD4CHELoader
from dronalize.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from dronalize.datasets.shared.presets import (
    lane_change_sampling,
    linear_resample,
    minimum_samples_screening,
    scenes_config,
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="ad4che",
    loader_factory=AD4CHELoader.from_loader_request,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=61,
            future_frames=150,
            sample_time=1 / 30,
            window_step=25,
            resample=linear_resample(up=1, down=3),
            lane_change=lane_change_sampling(required_lane_changes=5, negative_keep_every=3),
        ),
        map=MapConfig(extraction=FullMapExtraction(), interpolation_distance=8),
        screening=minimum_samples_screening(6, prediction_frame=60),
    ),
    native_schema=AD4CHELoader.native_trajectory_schema(),
    split_support=DatasetSplitSupport(scene=True, source=True, time_block=True),
    feature_support=DatasetFeatureSupport(map=True, lane_change_sampling=True),
)
