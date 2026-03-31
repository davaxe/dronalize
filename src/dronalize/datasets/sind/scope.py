from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.datasets.sind.loader import SindLoader as _Loader
from dronalize.datasets.sind.maps.builder import SindMapBuilder as _MapBuilder
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.maps.config import MapConfig

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

    from dronalize.core.maps.graph import MapGraph
    from dronalize.processing.maps.resolver import MapKey


@contextmanager
def sind_execution_scope(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig | None
) -> Generator[None, None, None]:
    """Prepare shared map state for a SinD processing run.

    Parameters
    ----------
    root : Path
        Root directory of the extracted SinD dataset.
    loader_config : LoaderConfig
        Unused by this hook. Included to match the standard dataset scope signature.
    map_config : MapConfig
        Map-building configuration. If maps are disabled, this scope only clears
        any previous shared-memory state.

    """
    _loader_config = loader_config
    if map_config is None:
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
            min_distance=map_config.min_distance, interp_distance=map_config.interp_distance
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
