from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.vod.loader import VodLoader as _Loader
from dronalize.datasets.vod.map.builder import VODMapBuilder as _MapBuilder

if TYPE_CHECKING:
    from dronalize.maps.graph import MapGraph


@contextmanager
def vod_execution_scope(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig
) -> Generator[None, None, None]:
    """Lifecycle context for the VOD dataset.

    Note that this is a context manager that will keep the map graph in shared
    memory for the duration of the context.

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

    map_dir = root / "maps" / "expansion" / "delft.json"
    builder = _MapBuilder.from_json_file(map_dir)
    map_graph: MapGraph = builder.build(
        min_distance=map_config.min_distance,
        interp_distance=map_config.interp_distance,
    )
    shm = map_graph.to_shared()

    _Loader.set_shared_memory(name=shm.name)
    yield
    shm.close()
    shm.unlink()
    _Loader.set_shared_memory()
