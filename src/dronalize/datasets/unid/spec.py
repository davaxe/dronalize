from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dronalize.config.models import (
    DatasetConfig,
    FullMapExtraction,
    MapConfig,
    ScenesConfig,
)
from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.shared.resources import open_named_shared_map_resources
from dronalize.datasets.shared.specs import (
    minimum_samples_screening,
    scenes_config,
    spline_resample,
)
from dronalize.datasets.unid.loader import UniDLoader
from dronalize.datasets.unid.maps.builder import UniDMapBuilder
from dronalize.processing.loading.resources import DatasetResources


@contextmanager
def open_unid_resources(
    root: Path, scenes: ScenesConfig, map_config: MapConfig | None
) -> Generator[DatasetResources, None, None]:
    """Build shared uniD maps once per run."""
    _ = scenes
    with open_named_shared_map_resources(
        map_config=map_config,
        named_paths=(
            (map_path.stem[len("location") :], map_path)
            for map_path in (root / "maps" / "lanelets").rglob("*.osm")
        ),
        build_map=lambda path, config: UniDMapBuilder(path).build(
            config.min_distance, config.interp_distance
        ),
    ) as resources:
        yield resources


DATASET_SPEC = DatasetSpec(
    name="unid",
    loader_factory=UniDLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(
            history_frames=50,
            future_frames=125,
            sample_time=1 / 25,
            window_step=25,
            resample=spline_resample(up=2),
        ),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=FullMapExtraction()),
    ),
    native_schema=UniDLoader.native_trajectory_schema(),
    resources_factory=open_unid_resources,
    has_map=True,
)
