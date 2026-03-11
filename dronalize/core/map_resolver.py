from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dronalize.config.map import MapConfig
    from dronalize.core.map_graph import MapGraph
    from dronalize.core.scene import Scene

MapKey = str | None
"""Lightweight map identifier stored on each scene.

`None` means "no map" or "use the default/only map".  A non-`None`
string is resolved by a `MapResolver` to produce a
`~dronalize.core.map_graph.MapGraph`.
"""


@runtime_checkable
class MapResolver(Protocol):
    """Protocol for resolving a :data:`MapKey` into a `MapGraph`.

    Implementations are expected to handle caching internally when
    appropriate (e.g. via `functools.lru_cache`).

    A resolver is typically obtained from the loader that produced the
    scenes via its `map_resolver()` method.
    """

    def __call__(
        self,
        scene: Scene,
        key: MapKey = None,
        map_config: MapConfig | None = None,
    ) -> MapGraph | None:
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
        map_config : MapConfig or None
            The configuration for the map.  If `None`, a default configuration
            is used.

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
        map_config: MapConfig | None = None,
    ) -> None:
        _ = scene, key, map_config  # Unuse

    _resolve.__name__ = "no_map"
    return _resolve
