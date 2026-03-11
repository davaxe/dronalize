from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dronalize.maps.graph import MapGraph

if TYPE_CHECKING:
    from collections.abc import Callable

    from dronalize.scene import Scene

MapKey = str | None
"""Lightweight map identifier stored on each scene.

`None` means "no map" or "use the default/only map". A non-`None` string is
resolved by a `MapResolver` to produce a `MapGraph`.

"""


@runtime_checkable
class MapResolver(Protocol):
    """Protocol for resolving a :data:`MapKey` into a `MapGraph`.

    Implementations are expected to handle caching internally when
    appropriate (e.g. via `functools.lru_cache`).

    A resolver is typically obtained from the loader that produced the
    scenes via its `map_resolver()` method.
    """

    def __call__(self, scene: Scene, key: MapKey) -> MapGraph | None:
        """Resolve *key* into a `MapGraph`, or `None`.

        Parameters
        ----------
        scene : Scene
            The scene for which to resolve the map.  This is provided for
            context and potential use in resolution, but resolvers are not
            required to use it.
        key : MapKey
            The map key to resolve.  `None` may be used for datasets
            that have a single shared map or no map at all.

        Returns
        -------
        MapGraph or None
            The resolved map graph, or `None` if no map is available
            for the given key.

        """
        ...


def no_map() -> MapResolver:
    """Create a resolver for datasets that have no map data.

    Returns
    -------
    MapResolver
        A resolver that always returns `None`.

    """

    def _resolve(
        scene: Scene,
        key: MapKey = None,
    ) -> None:
        _ = scene, key  # Unused

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
    f : Callable[[MapGraph], MapGraph] | None
        A function to apply to the map graph before returning it.
        If `None`, the map graph is returned as-is.

    Returns
    -------
    MapResolver
        A resolver that resolves the map from shared memory.

    """

    def _resolve(scene: Scene, key: MapKey) -> MapGraph | None:
        name = shared_name.get(key) if isinstance(shared_name, dict) else shared_name
        if name is None:
            return None

        with MapGraph.from_shared(name) as map_graph:
            if f is None:
                return map_graph.copy()

            return f(scene, map_graph).copy()

    _resolve.__name__ = "shared_map"
    return _resolve
