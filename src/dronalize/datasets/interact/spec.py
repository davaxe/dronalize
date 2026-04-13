from dronalize.config.models import DatasetConfig
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.interact.loader import InteractionLoader, InteractionLoaderOptions
from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config

DATASET_SPEC = DatasetSpec(
    name="interact",
    loader_factory=InteractionLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=10, future_frames=30, sample_time=0.1),
        screening=minimum_samples_screening(2),
        dataset=InteractionLoaderOptions().model_dump(),
    ),
    native_schema=InteractionLoader.native_trajectory_schema(),
    native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST),
    dataset_options_model=InteractionLoaderOptions,
)
