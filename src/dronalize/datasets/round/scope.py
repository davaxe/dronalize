from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.datasets.round.loader import RounDLoader as _Loader
from dronalize.datasets.round.maps.builder import RounDMapBuilder as _MapBuilder
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.maps.config import MapConfig

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

    from dronalize.core.maps.graph import MapGraph
    from dronalize.processing.maps.resolver import MapKey


@contextmanager
def round_execution_scope(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig
) -> Generator[None, None, None]:
    """Prepare shared map state for a rounD processing run.

    Parameters
    ----------
    root : Path
        Root directory of the extracted rounD dataset.
    loader_config : LoaderConfig
        Unused by this hook. Included to match the standard dataset scope signature.
    map_config : MapConfig
        Map-building configuration. If maps are disabled, this scope only clears
        any previous shared-memory state.

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
            min_distance=map_config.min_distance, interp_distance=map_config.interp_distance
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
