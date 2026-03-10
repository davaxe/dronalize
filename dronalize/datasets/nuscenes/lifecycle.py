from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.core.datatypes.loader_config import LoaderConfig
from dronalize.core.datatypes.map_config import MapConfig
from dronalize.datasets.nuscenes.loader import NuScenesLoader
from dronalize.datasets.nuscenes.map.graph_builder import NuScenesMapGraphBuilder

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

    from dronalize.core.datatypes.map_graph import MapGraph
    from dronalize.core.datatypes.map_resolver import MapKey


@contextmanager
def nuscenes_lifecylce_context(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig
) -> Generator[None, None, None]:
    """Lifecycle context for the Nuscenes dataset.

    This will build the map graph from the raw map files and store it in shared
    memory for use by the scene loader. The shared memory is cleaned up after
    processing is complete.

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
        NuScenesLoader.set_shared_memory()
        yield
        return

    map_dir = root / "nuScenes-map-expansion-v1.3" / "expansion"

    shm: list[SharedMemory] = []
    name_mapping: dict[MapKey, str] = {}
    for path in map_dir.glob("*.json"):
        builder = NuScenesMapGraphBuilder.from_json_file(path)
        map_graph: MapGraph = builder.build(
            min_distance=map_config.min_distance,
            interp_distance=map_config.interp_distance,
        )
        shm.append(map_graph.to_shared())
        name_mapping[path.stem] = shm[-1].name

    NuScenesLoader.set_shared_memory(mappings=name_mapping)
    yield
    for sm in shm:
        sm.close()
        sm.unlink()
    NuScenesLoader.set_shared_memory()
