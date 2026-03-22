from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.ind.loader import InDLoader as _Loader
from dronalize.datasets.ind.map.builder import InDMapBuilder as _MapBuilder

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

    from dronalize.maps.graph import MapGraph
    from dronalize.maps.resolver import MapKey


@contextmanager
def ind_execution_scope(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig,
) -> Generator[None, None, None]:
    """Lifecycle context for the inD dataset.

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

    shm: list[SharedMemory] = []
    mappings: dict[MapKey, str] = {}
    map_dir = root / "maps" / "lanelets"
    for map_path in map_dir.rglob("*.osm"):
        number = map_path.stem[len("location") :]
        builder = _MapBuilder(map_path)
        map_graph: MapGraph = builder.build(
            min_distance=map_config.min_distance,
            interp_distance=map_config.interp_distance,
        )
        shm.append(map_graph.to_shared())
        mappings[number] = shm[-1].name

    _Loader.set_shared_memory(mappings=mappings)
    try:
        yield
    finally:
        for shm_i in shm:
            shm_i.close()
            shm_i.unlink()
        _Loader.set_shared_memory()
