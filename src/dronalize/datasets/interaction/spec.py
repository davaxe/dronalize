from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dronalize.config.models import DatasetConfig
from dronalize.config.models.map import MapConfig
from dronalize.config.models.scenes import ScenesConfig
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.interaction.loader import InteractionLoader
from dronalize.datasets.interaction.maps.builder import InteractionMapBuilder
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.resources import open_named_shared_map_resources
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config
from dronalize.processing.loading.resources import DatasetResources


@contextmanager
def open_interaction_resources(
    root: Path, scenes: ScenesConfig, map_config: MapConfig | None
) -> Generator[DatasetResources, None, None]:
    """Build shared SinD maps once per run."""
    _ = scenes
    if map_config is None:
        yield DatasetResources()
        return

    paths: list[Path] = list((root / "maps").glob("*.osm_xy"))
    named_paths = [(path.stem, path) for path in paths if path.is_file()]
    with open_named_shared_map_resources(
        map_config=map_config,
        named_paths=named_paths,
        build_map=lambda path, config: InteractionMapBuilder(path).build(
            config.min_distance, config.interp_distance
        ),
    ) as resources:
        yield resources


DATASET_SPEC = DatasetSpec(
    name="interaction",
    loader_factory=InteractionLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=10, future_frames=30, sample_time=0.1),
        screening=minimum_samples_screening(2),
    ),
    native_schema=InteractionLoader.native_trajectory_schema(),
    native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST),
    resources_factory=open_interaction_resources,
    has_map=True,
    split_support=DatasetSplitSupport(scene=True),
)
