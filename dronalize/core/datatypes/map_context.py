from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

if TYPE_CHECKING:
    from dronalize.core.datatypes.map_graph import MapGraph

ID = TypeVar("ID", bound=Hashable)
T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class NoMap:
    """The dataset has no map."""

    tag: Literal["NoMap"] = "NoMap"


@dataclass(slots=True, frozen=True)
class SharedMap:
    """One fixed map covers all scenes in the dataset."""

    tag: Literal["SharedMap"] = "SharedMap"


@dataclass(slots=True, frozen=True)
class ReferencedMap:
    """The map is identified by a key that a registry/builder can resolve."""

    map_id: str
    tag: Literal["ReferencedMap"] = "ReferencedMap"


@dataclass(slots=True, frozen=True)
class LoadedMap:
    """The map is fully loaded and included in the context."""

    map: MapGraph
    tag: Literal["LoadedMap"] = "LoadedMap"


MapContext = NoMap | SharedMap | ReferencedMap | LoadedMap
