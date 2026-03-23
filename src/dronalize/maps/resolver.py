from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from dronalize.maps.graph import MapGraph

if TYPE_CHECKING:
    from dronalize.scene._scene import Scene

MapKey = str | None
MapResolver = Callable[["Scene"], MapGraph | None]

__all__ = ["MapKey", "MapResolver", "no_map", "shared_map"]


def no_map() -> MapResolver:
    """Create a resolver for datasets that have no map data.

    Returns
    -------
    MapResolver
        A resolver that always returns `None`.

    """

    def _resolve(_scene: Scene) -> None:
        return None

    _resolve.__name__ = "no_map"
    return _resolve


def shared_map(
    shared_name: dict[MapKey, str] | str,
    f: Callable[[Scene, MapGraph], MapGraph] | None = None,
) -> MapResolver:
    """Create a resolver that resolves a map from shared memory.

    Parameters
    ----------
    shared_name : dict[MapKey, str] | str
        The name of the shared memory block to use for the map.
        If a `dict`, the value is looked up by `map_key`.
    f : Callable[[Scene, MapGraph], MapGraph] | None
        A function to apply to the map graph before returning it.
        If `None`, the map graph is returned as-is.

    Returns
    -------
    MapResolver
        A resolver that resolves the map from shared memory.

    """

    def _resolve(scene: Scene) -> MapGraph | None:
        name = shared_name.get(scene.map_key) if isinstance(shared_name, dict) else shared_name
        if name is None:
            return None

        with MapGraph.from_shared(name) as map_graph:
            if f is None:
                return map_graph.copy()

            return f(scene, map_graph).copy()

    _resolve.__name__ = "shared_map"
    return _resolve
