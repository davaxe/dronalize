from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.us101.graph_builder import US101GraphBuilder as _GraphBuilder
from dronalize.datasets.us101.loader import US101Loader as _Loader

if TYPE_CHECKING:
    from dronalize.core.map_graph import MapGraph


@contextmanager
def us101_lifecycle_context(
    root: Path, loader_config: LoaderConfig, map_config: MapConfig
) -> Generator[None, None, None]:
    """Lifecycle context for the us101 dataset.

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

    builder = _GraphBuilder(root)
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
