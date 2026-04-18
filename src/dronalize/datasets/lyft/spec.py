from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dronalize.config.models import DatasetConfig, MapConfig, SceneExtentExtraction, ScenesConfig
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.lyft.loader import LyftLoader, LyftLoaderOptions
from dronalize.datasets.lyft.maps.builder import LyftMapBuilder
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.resources import open_single_shared_map_resource
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config
from dronalize.processing.loading.resources import DatasetResources


@contextmanager
def open_lyft_resources(
    root: Path, scenes: ScenesConfig, map_config: MapConfig | None
) -> Generator[DatasetResources, None, None]:
    """Build the shared Lyft map once per run."""
    _ = scenes
    if map_config is None:
        yield DatasetResources()
        return
    with open_single_shared_map_resource(
        map_config=map_config,
        map_path=root / "semantic_map" / "semantic_map.pb",
        build_map=lambda _, config: LyftMapBuilder.from_files(
            root / "semantic_map" / "semantic_map.pb", root / "semantic_map" / "meta.json"
        ).build(config.min_distance, config.interp_distance),
    ) as resources:
        yield resources


DATASET_SPEC = DatasetSpec(
    name="lyft",
    loader_factory=LyftLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=20, future_frames=50, sample_time=0.1, window_step=20),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=SceneExtentExtraction()),
        dataset=LyftLoaderOptions().model_dump(),
    ),
    native_schema=LyftLoader.native_trajectory_schema(),
    native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL),
    dataset_options_model=LyftLoaderOptions,
    resources_factory=open_lyft_resources,
    has_map=True,
    split_support=DatasetSplitSupport(scene=True),
)
