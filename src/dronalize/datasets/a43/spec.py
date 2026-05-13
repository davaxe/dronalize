from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig
from dronalize.datasets.a43.loader import A43Loader, A43LoaderOptions
from dronalize.datasets.registry import DatasetFeatureSupport, DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config

DATASET_SPEC = DatasetSpec(
    name="a43",
    loader_factory=A43Loader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=20, future_frames=50, sample_time=0.1, window_step=25),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=FullMapExtraction()),
    ),
    native_schema=A43Loader.native_trajectory_schema(),
    feature_support=DatasetFeatureSupport(map=True),
    loader_options_model=A43LoaderOptions,
    split_support=DatasetSplitSupport(scene=True, time_block=True),
)
