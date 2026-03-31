from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.datasets.lyft.loader import LyftLoader
from dronalize.datasets.lyft.maps.builder import LyftMapBuilder
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.maps.config import MapConfig

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

    from dronalize.core.maps.graph import MapGraph


@contextmanager
def lyft_execution_scope(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig | None
) -> Generator[None, None, None]:
    """Prepare shared map state for a Lyft Level 5 processing run.

    Parameters
    ----------
    root : Path
        Root directory of the extracted Lyft Level 5 dataset.
    loader_config : LoaderConfig
        Unused by this hook. Included to match the standard dataset scope signature.
    map_config : MapConfig
        Map-building configuration. If maps are disabled, this scope only clears
        any previous shared-memory state.

    """
    _loader_config = loader_config
    if map_config is None:
        LyftLoader.set_shared_memory()
        yield
        return

    map_path = root / "semantic_map" / "semantic_map.pb"
    meta_json_path = root / "semantic_map" / "meta.json"
    builder = LyftMapBuilder.from_files(map_path, meta_json_path)
    map_graph: MapGraph = builder.build(
        min_distance=map_config.min_distance, interp_distance=map_config.interp_distance
    )
    shm: SharedMemory = map_graph.to_shared()
    LyftLoader.set_shared_memory(shm.name)
    yield
    shm.close()
    shm.unlink()
    LyftLoader.set_shared_memory()
