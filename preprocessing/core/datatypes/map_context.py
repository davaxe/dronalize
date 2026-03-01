from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from preprocessing.core.datatypes.map_graph import MapGraph

ID = TypeVar("ID", bound=Hashable)


@dataclass(slots=True, frozen=True)
class Implicit:
    """Indicates an implicit map context; map information is known without any additional info.

    Commonly this is the case when there is only one possible map.
    """

    tag: Literal["Implicit"] = "Implicit"


@dataclass(slots=True, frozen=True)
class Explicit:
    """Indicates an explicit map context; map information is provided explicitly for the scene."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata required to determine the map."""
    tag: Literal["Explicit"] = "Explicit"

    def __init__(self, **metadata: Any) -> None:  # noqa: ANN401
        """Initialize with the given identifier and metadata."""
        object.__setattr__(self, "metadata", metadata)


@dataclass(slots=True, frozen=True)
class Loaded:
    """Indicates a loaded map context; the map graph is directly provided."""

    map: MapGraph
    """The map graph directly provided for the scene."""
    tag: Literal["Loaded"] = "Loaded"


@dataclass(slots=True, frozen=True)
class NoMap:
    """Indicates no map is available for the scene; map information is not known or not applicable."""

    tag: Literal["NoMap"] = "NoMap"


MapContext = Implicit | Explicit | Loaded | NoMap
