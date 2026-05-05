from dronalize.config.models import DatasetConfig, MapConfig, TrajectoryBufferExtraction
from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.datasets.lyft.loader import LyftLoader, LyftLoaderOptions
from dronalize.datasets.lyft.maps.builder import LyftMapBuilder
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.resources import single_shared_map_resource_factory
from dronalize.datasets.shared.specs import (
    combine_screenings,
    exclude_category_screening,
    minimum_samples_screening,
    scenes_config,
)

_open_lyft_resources = single_shared_map_resource_factory(
    map_path=lambda root: root / "semantic_map" / "semantic_map.pb",
    build_map=lambda path, config: LyftMapBuilder.from_files(
        path, path.with_name("meta.json")
    ).build(config.min_distance, config.interp_distance),
)


DATASET_SPEC = DatasetSpec(
    name="lyft",
    loader_factory=LyftLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=20, future_frames=50, sample_time=0.1, window_step=20),
        screening=combine_screenings(
            minimum_samples_screening(2), exclude_category_screening(AgentCategory.UNKNOWN)
        ),
        map=MapConfig(extraction=TrajectoryBufferExtraction(radius=25)),
        dataset=LyftLoaderOptions().model_dump(),
    ),
    native_schema=LyftLoader.native_trajectory_schema(),
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL),
    dataset_options_model=LyftLoaderOptions,
    resources_factory=_open_lyft_resources,
    has_map=True,
    split_support=DatasetSplitSupport(scene=True),
)
