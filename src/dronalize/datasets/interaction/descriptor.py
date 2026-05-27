from pathlib import Path

from dronalize.config.models.dataset import DatasetConfig
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.interaction.loader import InteractionLoader
from dronalize.datasets.interaction.maps import InteractionMapBuilder
from dronalize.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from dronalize.datasets.shared.presets import (
    minimum_samples_screening,
    scenes_config,
    temporal_support,
)
from dronalize.datasets.shared.resources import named_shared_map_resources_factory


def _interaction_map_paths(root: Path) -> list[tuple[str, Path]]:
    paths: list[Path] = list((root / "maps").glob("*.osm_xy"))
    return [(path.stem, path) for path in paths if path.is_file()]


_open_interaction_resources = named_shared_map_resources_factory(
    named_paths=_interaction_map_paths,
    build_map=lambda path, config: InteractionMapBuilder(path).build(
        config.min_distance, config.interpolation_distance
    ),
)


DATASET_DESCRIPTOR = DatasetDescriptor(
    name="interaction",
    loader_factory=InteractionLoader.from_loader_request,
    default_config=DatasetConfig(
        scenes=scenes_config(horizon_frames=40, default_observation_length=10, sample_time=0.1),
        screening=minimum_samples_screening(2, required_frame=9),
    ),
    native_schema=InteractionLoader.native_trajectory_schema(),
    supported_native_splits=(DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST),
    resources_factory=_open_interaction_resources,
    feature_support=DatasetFeatureSupport(map=True),
    split_support=DatasetSplitSupport(scene=True),
    temporal_support=temporal_support(
        source_unit="case", min_frames=10, max_frames=40, enabled_by_default=False
    ),
)
