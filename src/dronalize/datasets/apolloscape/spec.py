from dronalize.config.models import DatasetConfig
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.apolloscape.loader import ApolloScapeLoader
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.specs import (
    minimum_samples_screening,
    scenes_config,
    spline_resample,
)

DATASET_SPEC = DatasetSpec(
    name="apolloscape",
    loader_factory=ApolloScapeLoader.unified_runtime_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=4,
            future_frames=6,
            sample_time=0.5,
            window_step=1,
            resample=spline_resample(up=5),
        ),
        screening=minimum_samples_screening(2),
    ),
    native_schema=ApolloScapeLoader.native_trajectory_schema(),
    native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL),
    split_support=DatasetSplitSupport(scene=True, source=True),
)
