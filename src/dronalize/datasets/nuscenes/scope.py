from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.datasets.nuscenes.loader import NuScenesLoader
from dronalize.datasets.nuscenes.maps.builder import NuScenesMapBuilder
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.maps.config import MapConfig

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

    from dronalize.core.maps.graph import MapGraph
    from dronalize.processing.maps.resolver import MapKey


@contextmanager
def nuscenes_execution_scope(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig | None
) -> Generator[None, None, None]:
    """Prepare shared map state for a nuScenes processing run.

    Parameters
    ----------
    root : Path
        Root directory of the extracted nuScenes dataset.
    loader_config : LoaderConfig
        Unused by this hook. Included to match the standard dataset scope signature.
    map_config : MapConfig
        Map-building configuration. If maps are disabled, this scope only clears
        any previous shared-memory state.

    """
    _loader_config = loader_config
    if map_config is None:
        NuScenesLoader.set_shared_memory()
        yield
        return

    map_dir = root / "nuScenes-map-expansion-v1.3" / "expansion"

    shm: list[SharedMemory] = []
    name_mapping: dict[MapKey, str] = {}
    for path in map_dir.glob("*.json"):
        builder = NuScenesMapBuilder.from_json_file(path)
        map_graph: MapGraph = builder.build(
            min_distance=map_config.min_distance, interp_distance=map_config.interp_distance
        )
        shm.append(map_graph.to_shared())
        name_mapping[path.stem] = shm[-1].name

    NuScenesLoader.set_shared_memory(mappings=name_mapping)
    yield
    for sm in shm:
        sm.close()
        sm.unlink()
    NuScenesLoader.set_shared_memory()
