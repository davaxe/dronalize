from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.sind.loader import SindLoader as _Loader
from dronalize.datasets.sind.map.builder import SindMapBuilder as _MapBuilder

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

    from dronalize.maps.graph import MapGraph
    from dronalize.maps.resolver import MapKey


@contextmanager
def sind_execution_scope(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig
) -> Generator[None, None, None]:
    """Lifecycle context for the SIND dataset.

    Parameters
    ----------
    root : Path
        Root directory of the dataset.
    loader_config : LoaderConfig
        The loader configuration for this dataset. Not used in this startup hook,
        but included in the signature for consistency with other datasets.
    map_config : MapConfig
        The map configuration for this dataset, which specifies parameters for
        building the map graph.

    """
    _loader_config = loader_config
    if not map_config.include_map:
        _Loader.set_shared_memory()
        yield
        return

    key_path: list[tuple[str, str]] = [
        ("changchun", "Changchun_Pudong.osm"),
        ("xian", "Xi'an_Shanglin.osm"),
        ("nr_ll2", "NR_ll2.osm"),
        ("map_relink_law_save", "map_relink_law_save.osm"),
    ]

    shm: list[SharedMemory] = []
    mappings: dict[MapKey, str] = {}
    map_dir = root / "maps"
    for key, path in key_path:
        map_path = map_dir / path
        builder = _MapBuilder(map_path)
        map_graph: MapGraph = builder.build(
            min_distance=map_config.min_distance,
            interp_distance=map_config.interp_distance,
        )
        shm.append(map_graph.to_shared())
        mappings[key] = shm[-1].name

    _Loader.set_shared_memory(mappings=mappings)
    try:
        yield
    finally:
        for shm_i in shm:
            shm_i.close()
            shm_i.unlink()
        _Loader.set_shared_memory()
