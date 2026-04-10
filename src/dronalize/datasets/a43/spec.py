from dronalize.config.sections import (
    DatasetConfig,
    FullMapExtraction,
    MapConfig,
)
from dronalize.datasets.a43.loader import A43Loader
from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config
from dronalize.processing.loading.loader import BlockSplitSupport

DATASET_SPEC = DatasetSpec(
    name="a43",
    loader_factory=A43Loader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=20, future_frames=50, sample_time=0.1, window_step=25
        ),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=FullMapExtraction()),
    ),
    native_schema=A43Loader.native_trajectory_schema(),
    has_map=True,
    time_split_support=BlockSplitSupport(),
)
