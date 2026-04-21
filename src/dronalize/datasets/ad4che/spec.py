from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig
from dronalize.datasets.ad4che.loader import AD4CHELoader
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.specs import (
    lane_change_sampling,
    minimum_samples_screening,
    resample_config,
    scenes_config,
)

DATASET_SPEC = DatasetSpec(
    name="ad4che",
    loader_factory=AD4CHELoader.unified_runtime_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=61,
            future_frames=150,
            sample_time=1 / 30,
            window_step=25,
            resample=resample_config(method="linear", up=1, down=3),
            lane_change=lane_change_sampling(required_lane_changes=5, negative_keep_every=3),
        ),
        map=MapConfig(extraction=FullMapExtraction()),
        screening=minimum_samples_screening(4),
    ),
    native_schema=AD4CHELoader.native_trajectory_schema(),
    split_support=DatasetSplitSupport(scene=True, source=True, time_block=True),
    has_map=True,
)
