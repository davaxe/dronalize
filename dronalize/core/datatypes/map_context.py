"""Map resolution for scenes.

A :data:`MapKey` is a lightweight, picklable identifier stored on each
`~dronalize.core.datatypes.scene.Scene`.  A `MapResolver`
turns that key into a concrete `~dronalize.core.datatypes.map_graph.MapGraph`
on demand.

Built-in resolver factories
----------------------------
* `no_map` — dataset has no map data at all.
* `fixed_map` — one shared map for the entire dataset, loaded lazily.
* `keyed_map` — maps looked up by string key, with LRU caching.
* `preloaded_map` — maps already built and stored in a dict.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from dronalize.core.datatypes.map_graph import MapGraph

MapKey = str | None
"""Lightweight map identifier stored on each scene.

`None` means "no map" or "use the default/only map".  A non-`None`
string is resolved by a `MapResolver` to produce a
`~dronalize.core.datatypes.map_graph.MapGraph`.
"""


@runtime_checkable
class MapResolver(Protocol):
    """Protocol for resolving a :data:`MapKey` into a `MapGraph`.

    Implementations are expected to handle caching internally when
    appropriate (e.g. via `functools.lru_cache`).

    A resolver is typically obtained from the loader that produced the
    scenes via its `map_resolver()` method.
    """

    def __call__(self, key: MapKey = None) -> MapGraph | None:
        """Resolve *key* into a `MapGraph`, or `None`.

        Parameters
        ----------
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


# ---------------------------------------------------------------------------
# Built-in resolver factories
# ---------------------------------------------------------------------------


def no_map() -> MapResolver:
    """Create a resolver for datasets that have no map data.

    Returns
    -------
    MapResolver
        A resolver that always returns `None`.

    Example
    -------
    ::

        def map_resolver(self) -> MapResolver:
            return no_map()

    """

    def _resolve(key: MapKey = None) -> None:  # noqa: ARG001
        return None

    _resolve.__name__ = "no_map"
    _resolve.__qualname__ = "no_map._resolve"
    return _resolve


def fixed_map(loader: Callable[[], MapGraph]) -> MapResolver:
    """Create a resolver for datasets with a single shared map.

    The map is loaded lazily on first access and cached for subsequent
    calls.

    Parameters
    ----------
    loader : Callable[[], MapGraph]
        A zero-argument callable that loads and returns the map graph.

    Returns
    -------
    MapResolver

    Example
    -------
    ::

        def map_resolver(self) -> MapResolver:
            return fixed_map(lambda: load_osm(self._map_path))

    """
    cached: list[MapGraph] = []  # mutable container for closure-based cache

    def _resolve(key: MapKey = None) -> MapGraph:  # noqa: ARG001
        if not cached:
            cached.append(loader())
        return cached[0]

    _resolve.__name__ = "fixed_map"
    _resolve.__qualname__ = "fixed_map._resolve"
    return _resolve


def keyed_map(
    loader: Callable[[str], MapGraph],
    *,
    maxsize: int = 16,
) -> MapResolver:
    """Create a resolver that looks up maps by string key with LRU caching.

    Parameters
    ----------
    loader : Callable[[str], MapGraph]
        A callable that takes a map key string and returns the
        corresponding `MapGraph`.
    maxsize : int, optional
        Maximum number of cached maps.  Defaults to 16.

    Returns
    -------
    MapResolver

    Example
    -------
    ::

        def map_resolver(self) -> MapResolver:
            return keyed_map(
                lambda key: load_osm(self._map_dir / key),
                maxsize=4,
            )

    """

    @lru_cache(maxsize=maxsize)
    def _cached_load(key: str) -> MapGraph:
        return loader(key)

    def _resolve(key: MapKey = None) -> MapGraph | None:
        if key is None:
            return None
        return _cached_load(key)

    _resolve.__name__ = "keyed_map"
    _resolve.__qualname__ = "keyed_map._resolve"
    return _resolve


def preloaded_map(maps: dict[str, MapGraph]) -> MapResolver:
    """Create a resolver backed by a pre-populated dictionary.

    This is useful when maps are built during ingestion (e.g. Waymo,
    where the map is embedded in the same proto as the trajectories).

    Parameters
    ----------
    maps : dict[str, MapGraph]
        A dictionary mapping keys to already-built map graphs.  The dict
        may be mutated externally (e.g. populated during ingestion) and
        the resolver will see the updates.

    Returns
    -------
    MapResolver

    Example
    -------
    ::

        self._maps: dict[str, MapGraph] = {}

        def map_resolver(self) -> MapResolver:
            return preloaded_map(self._maps)

    """

    def _resolve(key: MapKey = None) -> MapGraph | None:
        if key is None:
            return None
        return maps.get(key)

    _resolve.__name__ = "preloaded_map"
    _resolve.__qualname__ = "preloaded_map._resolve"
    return _resolve
