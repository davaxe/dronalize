from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dronalize.config.models import DatasetConfig, FullMapExtraction, MapConfig, ScenesConfig
from dronalize.datasets.registry import DatasetSpec, DatasetSplitSupport
from dronalize.datasets.shared.resources import open_named_shared_map_resources
from dronalize.datasets.shared.specs import minimum_samples_screening, scenes_config
from dronalize.datasets.sind.loader import SindLoader
from dronalize.datasets.sind.maps.builder import SindMapBuilder
from dronalize.processing.loading.resources import DatasetResources


@contextmanager
def open_sind_resources(
    root: Path, scenes: ScenesConfig, map_config: MapConfig | None
) -> Generator[DatasetResources, None, None]:
    """Build shared SinD maps once per run."""
    _ = scenes
    if map_config is None:
        yield DatasetResources()
        return
    with open_named_shared_map_resources(
        map_config=map_config,
        named_paths=(
            ("Changchun", root / "Changchun" / "Changchun_Pudong.osm"),
            ("Xi'an", root / "Xi'an" / "Xi'an_Shanglin.osm"),
            ("Chongqing", root / "Chongqing" / "NR_ll2.osm"),
            ("Tianjin", root / "Tianjin" / "map_relink_law_save.osm"),
        ),
        build_map=lambda path, config: SindMapBuilder(path).build(
            config.min_distance, config.interp_distance
        ),
    ) as resources:
        yield resources


DATASET_SPEC = DatasetSpec(
    name="sind",
    loader_factory=SindLoader.unified_runtime_factory,
    default_config=DatasetConfig(
        scenes=scenes_config(history_frames=20, future_frames=50, sample_time=0.1, window_step=25),
        screening=minimum_samples_screening(2),
        map=MapConfig(extraction=FullMapExtraction()),
    ),
    native_schema=SindLoader.native_trajectory_schema(),
    resources_factory=open_sind_resources,
    has_map=True,
    split_support=DatasetSplitSupport(scene=True, source=True),
)
