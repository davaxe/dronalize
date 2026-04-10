from dronalize.config.sections import DatasetConfig, FullMapExtraction, MapConfig
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.argoverse2.loader import Argoverse2Loader, Argoverse2LoaderOptions
from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

DATASET_SPEC = DatasetSpec(
    name="argoverse2",
    loader_factory=Argoverse2Loader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=50, future_frames=60, sample_time=0.1),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=FullMapExtraction()),
        dataset=Argoverse2LoaderOptions().model_dump(),
    ),
    native_schema=Argoverse2Loader.native_trajectory_schema(),
    native_splits=_NATIVE_SPLITS,
    dataset_options_model=Argoverse2LoaderOptions,
    has_map=True,
)
