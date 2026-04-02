"""Execution-scope helpers for the Argoverse 1 dataset."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.datasets.argoverse1.loader import Argoverse1Loader
from dronalize.datasets.argoverse1.maps.builder import Argoverse1MapBuilder
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.maps.config import MapConfig

if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory

    from dronalize.core.maps.graph import MapGraph
    from dronalize.processing.maps.resolver import MapKey


@contextmanager
def argoverse1_execution_scope(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig | None
) -> Generator[None, None, None]:
    """Prepare shared map state for an Argoverse 1 processing run.

    Parameters
    ----------
    root : Path
        Root directory of the extracted Argoverse 1 dataset.
    loader_config : LoaderConfig
        Unused by this hook. Included to match the standard dataset scope signature.
    map_config : MapConfig | None
        Map-building configuration. If maps are disabled, this scope only clears
        any previous shared-memory state.

    """
    _loader_config = loader_config
    if map_config is None:
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
            min_distance=map_config.min_distance, interp_distance=map_config.interp_distance
        )
        shm.append(map_graph.to_shared())
        mappings[key] = shm[-1].name

    Argoverse1Loader.set_shared_memory(mappings=mappings)
    yield
    for shm_i in shm:
        shm_i.close()
        shm_i.unlink()
    Argoverse1Loader.set_shared_memory()
