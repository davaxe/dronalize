from pathlib import Path

from dronalize.config.models import DatasetConfig
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.interaction.loader import InteractionLoader
from dronalize.datasets.interaction.maps.builder import InteractionMapBuilder
from dronalize.datasets.registry import DatasetFeatureSupport, DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.resources import named_shared_map_resources_factory
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config


def _interaction_map_paths(root: Path) -> list[tuple[str, Path]]:
    paths: list[Path] = list((root / "maps").glob("*.osm_xy"))
    return [(path.stem, path) for path in paths if path.is_file()]


_open_interaction_resources = named_shared_map_resources_factory(
    named_paths=_interaction_map_paths,
    build_map=lambda path, config: InteractionMapBuilder(path).build(
        config.min_distance, config.interp_distance
    ),
)


DATASET_SPEC = DatasetSpec(
    name="interaction",
    loader_factory=InteractionLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=10, future_frames=30, sample_time=0.1),
        screening=minimum_samples_screening(2),
    ),
    native_schema=InteractionLoader.native_trajectory_schema(),
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST),
    resources_factory=_open_interaction_resources,
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True),
)
