from dronalize.config.models.dataset import DatasetConfig
from dronalize.config.models.map import MapConfig, TrajectoryBufferExtraction
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from dronalize.datasets.shared.presets import (
    minimum_samples_screening,
    scenes_config,
    temporal_support,
)
from dronalize.datasets.waymo.loader import WaymoLoader

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="waymo",
    loader_factory=WaymoLoader.from_loader_request,
    default_config=DatasetConfig(
        scenes=scenes_config(horizon_frames=91, default_observation_length=11, sample_time=0.1),
        screening=minimum_samples_screening(2, required_frame=10),
        map=MapConfig(extraction=TrajectoryBufferExtraction(radius=25)),
    ),
    native_schema=WaymoLoader.native_trajectory_schema(),
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST),
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True),
    temporal_support=temporal_support(
        source_unit="scenario",
        min_frames=11,
        max_frames=91,
        enabled_by_default=False,
        confidence="documented",
    ),
)
