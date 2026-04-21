from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig
from dronalize.datasets.highd.loader import HighDLoader
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.specs import (
    lane_change_sampling,
    minimum_samples_screening,
    scenes_config,
    spline_resample,
)

DATASET_SPEC = DatasetSpec(
    name="highd",
    loader_factory=HighDLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=50,
            future_frames=125,
            sample_time=1 / 25,
            window_step=25,
            resample=spline_resample(up=2),
            lane_change=lane_change_sampling(required_lane_changes=3, negative_keep_every=3),
        ),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=FullMapExtraction(), interp_distance=10),
    ),
    native_schema=HighDLoader.native_trajectory_schema(),
    has_map=True,
    split_support=DatasetSplitSupport(scene=True, source=True),
)
