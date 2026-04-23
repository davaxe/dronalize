from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig
from dronalize.datasets.levelx.loader import ExiDLoader, HighDLoader, StandardLevelXLoader
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport, LoaderFactory
from dronalize.datasets.shared.osm_builder import OSMMapBuilder
from dronalize.datasets.shared.resources import named_shared_map_resources_factory
from dronalize.datasets.shared.specs import (
    lane_change_sampling,
    linear_resample,
    minimum_samples_screening,
    scenes_config,
)

_open_levelx_osm_resources = named_shared_map_resources_factory(
    named_paths=lambda root: (
        (map_path.stem[len("location") :], map_path)
        for map_path in (root / "maps" / "lanelets").rglob("*.osm")
    ),
    build_map=lambda path, config: OSMMapBuilder(path).build(
        config.min_distance, config.interp_distance
    ),
)


def _levelx_config(*, lane_change: bool = False) -> DatasetConfig:
    return DatasetConfig(
        scenes=scenes_config(
            history_frames=50,
            future_frames=125,
            sample_time=1 / 25,
            window_step=25,
            resample=linear_resample(up=2, down=5),
            lane_change=lane_change_sampling(required_lane_changes=3, negative_keep_every=3)
            if lane_change
            else None,
        ),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=FullMapExtraction()),
    )


def _highd_config() -> DatasetConfig:
    return DatasetConfig(
        scenes=scenes_config(
            history_frames=50,
            future_frames=125,
            sample_time=1 / 25,
            window_step=25,
            resample=linear_resample(up=2, down=5),
            lane_change=lane_change_sampling(required_lane_changes=3, negative_keep_every=3),
        ),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=FullMapExtraction(), interp_distance=10),
    )


def _levelx_spec(
    name: str, loader_factory: LoaderFactory, default_config: DatasetConfig
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        loader_factory=loader_factory,
        default_config=default_config,
        native_schema=StandardLevelXLoader.native_trajectory_schema(),
        resources_factory=_open_levelx_osm_resources,
        has_map=True,
        split_support=DatasetSplitSupport(scene=True, source=True),
    )


DATASET_SPECS = {
    "highd": DatasetSpec(
        name="highd",
        loader_factory=HighDLoader.unified_factory,
        default_config=_highd_config(),
        native_schema=HighDLoader.native_trajectory_schema(),
        has_map=True,
        split_support=DatasetSplitSupport(scene=True, source=True),
    ),
    "ind": _levelx_spec("ind", StandardLevelXLoader.unified_factory, _levelx_config()),
    "exid": _levelx_spec("exid", ExiDLoader.unified_factory, _levelx_config(lane_change=True)),
    "round": _levelx_spec("round", StandardLevelXLoader.unified_factory, _levelx_config()),
    "unid": _levelx_spec("unid", StandardLevelXLoader.unified_factory, _levelx_config()),
}
