from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.core.datatypes.loader_config import LoaderConfig
from dronalize.core.datatypes.map_config import MapConfig
from dronalize.datasets.lyft.loader import LyftLoader
from dronalize.datasets.lyft.map.graph_builder import LyftMapGraphBuilder

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

    from dronalize.core.datatypes.map_graph import MapGraph


@contextmanager
def lyft_lifecylce_context(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig
) -> Generator[None, None, None]:
    """Lifecycle context for the Lyft dataset.

    This will build the map graph from the raw map files and store it in shared
    memory for use by the scene loader. The shared memory is cleaned up after
    processing is complete.

    Note that this is a context manager that will keep the map graph in shared
    memory for the duration of the context.

    Parameters
    ----------
    root : Path
        Root directory of the dataset, which contains the raw map files.
    loader_config : LoaderConfig
        The loader configuration for this dataset. Not used in this startup hook,
        but included in the signature for consistency with other datasets.
    map_config : MapConfig
        The map configuration for this dataset, which specifies parameters for
        building the map graph. If `map_config.include_map` is False, this startup
        hook will do nothing.

    """
    print("Setting up Lyft dataset lifecycle context...")
    _loader_config = loader_config
    if not map_config.include_map:
        LyftLoader.set_shared_memory()
        yield
        return

    map_path = root / "semantic_map" / "semantic_map.pb"
    meta_json_path = root / "semantic_map" / "meta.json"
    builder = LyftMapGraphBuilder.from_files(map_path, meta_json_path)
    map_graph: MapGraph = builder.build(
        min_distance=map_config.min_distance, interp_distance=map_config.interp_distance
    )
    shm: SharedMemory = map_graph.to_shared()
    LyftLoader.set_shared_memory(shm.name)
    yield
    shm.close()
    shm.unlink()
    print("Cleaned up Lyft dataset lifecycle context.")
    LyftLoader.set_shared_memory()
