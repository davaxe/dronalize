from dronalize.config.models import DatasetConfig, MapConfig, SceneExtentExtraction
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.argoverse1.loader import Argoverse1Loader, Argoverse1LoaderOptions
from dronalize.datasets.argoverse1.maps.builder import Argoverse1MapBuilder
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.resources import named_shared_map_resources_factory
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

_open_argoverse1_resources = named_shared_map_resources_factory(
    named_paths=lambda root: (
        ("MIA", root / "hd_maps" / "map_files" / "pruned_argoverse_MIA_10316_vector_map.xml"),
        ("PIT", root / "hd_maps" / "map_files" / "pruned_argoverse_PIT_10314_vector_map.xml"),
    ),
    build_map=lambda path, config: Argoverse1MapBuilder.from_xml_file(path).build(
        config.min_distance, config.interp_distance
    ),
)


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
    supported_native_splits=_NATIVE_SPLITS,
    dataset_options_model=Argoverse1LoaderOptions,
    resources_factory=_open_argoverse1_resources,
    split_support=DatasetSplitSupport(scene=True),
    has_map=True,
)
