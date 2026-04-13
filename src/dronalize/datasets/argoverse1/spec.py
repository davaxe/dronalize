from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dronalize.config.models import (
    DatasetConfig,
    MapConfig,
    SceneExtentExtraction,
    ScenesConfig,
)
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.argoverse1.loader import Argoverse1Loader, Argoverse1LoaderOptions
from dronalize.datasets.argoverse1.maps.builder import Argoverse1MapBuilder
from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.shared.resources import open_named_shared_map_resources
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config
from dronalize.processing.loading.resources import DatasetResources

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


@contextmanager
def open_argoverse1_resources(
    root: Path, scenes: ScenesConfig, map_config: MapConfig | None
) -> Generator[DatasetResources, None, None]:
    """Build shared Argoverse 1 maps once per run."""
    _ = scenes
    with open_named_shared_map_resources(
        map_config=map_config,
        named_paths=(
            (
                "MIA",
                root / "hd_maps" / "map_files" / "pruned_argoverse_MIA_10316_vector_map.xml",
            ),
            (
                "PIT",
                root / "hd_maps" / "map_files" / "pruned_argoverse_PIT_10314_vector_map.xml",
            ),
        ),
        build_map=lambda path, config: Argoverse1MapBuilder.from_xml_file(path).build(
            config.min_distance, config.interp_distance
        ),
    ) as resources:
        yield resources


DATASET_SPEC = DatasetSpec(
    name="argoverse1",
    loader_factory=Argoverse1Loader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=20, future_frames=30, sample_time=0.1),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=SceneExtentExtraction()),
        dataset=Argoverse1LoaderOptions().model_dump(),
    ),
    native_schema=Argoverse1Loader.native_trajectory_schema(),
    native_splits=_NATIVE_SPLITS,
    dataset_options_model=Argoverse1LoaderOptions,
    resources_factory=open_argoverse1_resources,
    has_map=True,
)
