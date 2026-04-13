"""Helpers for resolving per-scene map graphs at runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from dronalize.core.maps import MapGraph

if TYPE_CHECKING:
    from dronalize.core.scene.model import Scene

MapKey = str | None
"""Stable identifier for a map associated with a scene or source."""
MapResolver = Callable[["Scene"], MapGraph | None]
"""Callable that materializes a `MapGraph` for a scene on demand."""

__all__ = ["MapKey", "MapResolver", "no_map", "shared_map"]


def no_map() -> MapResolver:
    """Create a resolver for datasets that do not expose map data.

    Returns
    -------
    MapResolver
        Resolver that always returns `None`.

    """

    def _resolve(_scene: Scene) -> None:
        return None

    _resolve.__name__ = "no_map"
    return _resolve


def shared_map(
    shared_name: dict[MapKey, str] | str, f: Callable[[Scene, MapGraph], MapGraph] | None = None
) -> MapResolver:
    """Create a resolver that materializes a scene map from shared memory.

    Parameters
    ----------
    shared_name : dict[MapKey, str] | str
        Shared-memory name or lookup table keyed by `scene.map_key`.
    f : Callable[[Scene, MapGraph], MapGraph] | None
        A function to apply to the map graph before returning it.
        If `None`, the map graph is returned as-is.

    Returns
    -------
    MapResolver
        Resolver that opens the shared-memory map, optionally applies `f`,
        and returns a detached copy.

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
