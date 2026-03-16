from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.argoverse1.loader import Argoverse1Loader
from dronalize.datasets.argoverse1.map.builder import Argoverse1MapBuilder

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

    from dronalize.maps.graph import MapGraph
    from dronalize.maps.resolver import MapKey


@contextmanager
def argoverse1_execution_scope(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig
) -> Generator[None, None, None]:
    """Lifecycle context for the argoverse1 dataset.

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
        Argoverse1Loader.set_shared_memory()
        yield
        return

    map_dir = root / "hd_maps" / "map_files"
    paths: list[tuple[str, Path]] = [
        ("MIA", map_dir / "pruned_argoverse_MIA_10316_vector_map.xml"),
        ("PIT", map_dir / "pruned_argoverse_PIT_10314_vector_map.xml"),
    ]
    shm: list[SharedMemory] = []
    mappings: dict[MapKey, str] = {}
    for key, path in paths:
        if not path.exists():
            msg = f"Expected map file not found: {path}"
            raise FileNotFoundError(msg)

        builder = Argoverse1MapBuilder.from_xml_file(path)
        map_graph: MapGraph = builder.build(
            min_distance=map_config.min_distance,
            interp_distance=map_config.interp_distance,
        )
        shm.append(map_graph.to_shared())
        mappings[key] = shm[-1].name

    Argoverse1Loader.set_shared_memory(mappings=mappings)
    yield
    for shm_i in shm:
        shm_i.close()
        shm_i.unlink()
    Argoverse1Loader.set_shared_memory()
