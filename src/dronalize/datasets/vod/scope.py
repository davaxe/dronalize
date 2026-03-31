from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.datasets.vod.loader import VodLoader as _Loader
from dronalize.datasets.vod.maps.builder import VODMapBuilder as _MapBuilder
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.maps.config import MapConfig

if TYPE_CHECKING:
    from dronalize.core.maps.graph import MapGraph


@contextmanager
def vod_execution_scope(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig | None
) -> Generator[None, None, None]:
    """Prepare shared map state for a VOD processing run.

    Parameters
    ----------
    root : Path
        Root directory of the extracted VOD dataset.
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

    map_dir = root / "maps" / "expansion" / "delft.json"
    builder = _MapBuilder.from_json_file(map_dir)
    map_graph: MapGraph = builder.build(
        min_distance=map_config.min_distance, interp_distance=map_config.interp_distance
    )
    shm = map_graph.to_shared()

    _Loader.set_shared_memory(name=shm.name)
    yield
    shm.close()
    shm.unlink()
    _Loader.set_shared_memory()
