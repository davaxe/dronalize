from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig
from dronalize.datasets.levelx.loader import ExiDLoader, HighDLoader, StandardLevelXLoader
from dronalize.datasets.levelx.maps import HighDMapProvider
from dronalize.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetSplitSupport,
)
from dronalize.datasets.shared.osm_builder import OSMMapBuilder
from dronalize.datasets.shared.presets import (
    lane_change_sampling,
    linear_resample,
    minimum_observations_screening,
    scenes_config,
    temporal_support,
)
from dronalize.datasets.shared.resources import (
    map_provider_resources_factory,
    named_shared_map_resources_factory,
)

_open_levelx_osm_resources = named_shared_map_resources_factory(
    named_paths=lambda root: (
        (map_path.stem[len("location") :], map_path)
        for map_path in (root / "maps" / "lanelets").rglob("*.osm")
    ),
    build_map=lambda path, config: OSMMapBuilder(path).build(
        config.min_distance, config.interpolation_distance
    ),
)


_open_highd_resources = map_provider_resources_factory(
    create=lambda _root, map_config: HighDMapProvider(map_config)
)


def _levelx_config(*, lane_change: bool = False) -> DatasetConfig:
    return DatasetConfig(
        scenes=scenes_config(
            horizon_frames=175,
            default_observation_length=50,
            sample_time=1 / 25,
            window_step=25,
            resample=linear_resample(up=2, down=5),
            lane_change=lane_change_sampling(required_lane_changes=3, negative_keep_every=3)
            if lane_change
            else None,
        ),
        screening=minimum_observations_screening(2, required_frame=49),
        map=MapConfig(extraction=FullMapExtraction()),
    )


def _highd_config() -> DatasetConfig:
    return DatasetConfig(
        scenes=scenes_config(
            horizon_frames=175,
            default_observation_length=50,
            sample_time=1 / 25,
            window_step=25,
            resample=linear_resample(up=2, down=5),
            lane_change=lane_change_sampling(required_lane_changes=3, negative_keep_every=3),
        ),
        screening=minimum_observations_screening(2, required_frame=49),
        map=MapConfig(extraction=FullMapExtraction(), interpolation_distance=10),
    )


def _levelx_spec(
    name: str,
    loader_cls: type[StandardLevelXLoader],
    default_config: DatasetConfig,
    *,
    lane_change_sampling_support: bool = False,
    min_frames: int,
    max_frames: int,
) -> DatasetDescriptor:
    return DatasetDescriptor(
        name=name,
        loader_cls=loader_cls,
        default_config=default_config,
        native_schema=StandardLevelXLoader.native_trajectory_schema(),
        resources_factory=_open_levelx_osm_resources,
        feature_support=DatasetFeatureSupport(
            map=True, lane_change_sampling=lane_change_sampling_support
        ),
        split_support=DatasetSplitSupport(scene=True, source=True, time_block=True),
        temporal_support=temporal_support(
            source_unit="recording",
            min_frames=min_frames,
            max_frames=max_frames,
            enabled_by_default=True,
        ),
    )


DATASET_DESCRIPTORS = {
    "highd": DatasetDescriptor(
        name="highd",
        loader_cls=HighDLoader,
        default_config=_highd_config(),
        native_schema=HighDLoader.native_trajectory_schema(),
        feature_support=DatasetFeatureSupport(map=True, lane_change_sampling=True),
        split_support=DatasetSplitSupport(scene=True, source=True),
        temporal_support=temporal_support(
            source_unit="recording", min_frames=9729, max_frames=31274, enabled_by_default=True
        ),
        resources_factory=_open_highd_resources,
    ),
    "ind": _levelx_spec(
        "ind", StandardLevelXLoader, _levelx_config(), min_frames=16192, max_frames=33207
    ),
    "exid": _levelx_spec(
        "exid",
        ExiDLoader,
        _levelx_config(lane_change=True),
        lane_change_sampling_support=True,
        min_frames=1332,
        max_frames=43534,
    ),
    "round": _levelx_spec(
        "round", StandardLevelXLoader, _levelx_config(), min_frames=11024, max_frames=31240
    ),
    "unid": _levelx_spec(
        "unid", StandardLevelXLoader, _levelx_config(), min_frames=8095, max_frames=28658
    ),
}
