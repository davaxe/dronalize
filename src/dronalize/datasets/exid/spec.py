from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dronalize.config.models import (
    DatasetConfig,
    FullMapExtraction,
    MapConfig,
    ScenesConfig,
)
from dronalize.datasets.exid.loader import ExiDLoader
from dronalize.datasets.exid.maps.builder import ExiDMapBuilder
from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.shared.resources import open_named_shared_map_resources
from dronalize.datasets.shared.specs import (
    lane_change_sampling,
    minimum_samples_screening,
    scenes_config,
    spline_resample,
)
from dronalize.processing.loading.resources import DatasetResources


@contextmanager
def open_exid_resources(
    root: Path, scenes: ScenesConfig, map_config: MapConfig | None
) -> Generator[DatasetResources, None, None]:
    """Build shared ExiD maps once per run."""
    _ = scenes
    with open_named_shared_map_resources(
        map_config=map_config,
        named_paths=(
            (map_path.stem[len("location") :], map_path)
            for map_path in (root / "maps" / "lanelets").rglob("*.osm")
        ),
        build_map=lambda path, config: ExiDMapBuilder(path).build(
            config.min_distance, config.interp_distance
        ),
    ) as resources:
        yield resources


DATASET_SPEC = DatasetSpec(
    name="exid",
    loader_factory=ExiDLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=50,
            future_frames=125,
            sample_time=1 / 25,
            window_step=25,
            resample=spline_resample(up=2),
            lane_change=lane_change_sampling(required_lane_changes=3, negative_keep_every=3),
        ),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=FullMapExtraction()),
    ),
    native_schema=ExiDLoader.native_trajectory_schema(),
    resources_factory=open_exid_resources,
    has_map=True,
)
