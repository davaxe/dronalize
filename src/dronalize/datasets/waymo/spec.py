from dronalize.config.models import DatasetConfig, MapConfig, SceneExtentExtraction
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config
from dronalize.datasets.waymo.loader import WaymoLoader

DATASET_SPEC = DatasetSpec(
    name="waymo",
    loader_factory=WaymoLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=11, future_frames=80, sample_time=0.1),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=SceneExtentExtraction()),
    ),
    native_schema=WaymoLoader.native_trajectory_schema(),
    native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST),
    has_map=True,
    split_support=DatasetSplitSupport(scene=True),
)
