from dronalize.config.models import DatasetConfig, MapConfig, TrajectoryBufferExtraction
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from dronalize.datasets.shared.presets import minimum_samples_screening, scenes_config
from dronalize.datasets.waymo.loader import WaymoLoader

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="waymo",
    loader_factory=WaymoLoader.from_loader_request,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=11, future_frames=80, sample_time=0.1),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=TrajectoryBufferExtraction(radius=25)),
    ),
    native_schema=WaymoLoader.native_trajectory_schema(),
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST),
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True),
)
