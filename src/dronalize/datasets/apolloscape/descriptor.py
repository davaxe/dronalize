from dronalize.config.models import DatasetConfig
from dronalize.datasets.apolloscape.loader import ApolloScapeLoader
from dronalize.datasets.registry import DatasetDescriptor, DatasetSplitSupport
from dronalize.datasets.shared.presets import (
    linear_resample,
    minimum_observations_screening,
    scenes_config,
    temporal_support,
)

DATASET_DESCRIPTOR = DatasetDescriptor(
    name="apolloscape",
    loader_cls=ApolloScapeLoader,
    default_config=DatasetConfig(
        scenes=scenes_config(
            horizon_frames=10,
            default_observation_length=4,
            sample_time=0.5,
            window_step=1,
            resample=linear_resample(up=5),
        ),
        screening=minimum_observations_screening(2, required_frame=3),
    ),
    native_schema=ApolloScapeLoader.native_trajectory_schema(),
    split_support=DatasetSplitSupport(scene=True, source=True),
    temporal_support=temporal_support(
        source_unit="recording", min_frames=37, max_frames=119, enabled_by_default=True
    ),
)
