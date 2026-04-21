from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig
from dronalize.datasets.opendd.loader import OpenDDLoader
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.specs import (
    minimum_samples_screening,
    resample_config,
    scenes_config,
)

DATASET_SPEC = DatasetSpec(
    name="opendd",
    loader_factory=OpenDDLoader.unified_runtime_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=60,
            future_frames=150,
            sample_time=1 / 30,
            window_step=75,
            resample=resample_config(method="linear", up=1, down=3),
        ),
        screening=minimum_samples_screening(6),
        map=MapConfig(extraction=FullMapExtraction()),
    ),
    native_schema=OpenDDLoader.native_trajectory_schema(),
    has_map=True,
    split_support=DatasetSplitSupport(scene=True, source=True),
)
