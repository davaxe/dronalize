from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

if TYPE_CHECKING:
    from dronalize.core.datatypes.map_graph import MapGraph

ID = TypeVar("ID", bound=Hashable)
T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class Implicit:
    """Indicates an implicit map context.

    Commonly this is the case when there is only one possible map.
    """

    tag: Literal["Implicit"] = "Implicit"


@dataclass(slots=True, frozen=True)
class Explicit:
    """Indicates an explicit map context."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata required to determine the map."""
    tag: Literal["Explicit"] = "Explicit"

    def __init__(self, **metadata: Any) -> None:  # noqa: ANN401
        """Initialize with keyword arguments stored as metadata."""
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "tag", "Explicit")

    # -- Convenience accessors -----------------------------------------------

    @overload
    def get(self, key: str) -> Any: ...  # noqa: ANN401

    @overload
    def get(self, key: str, default: T) -> T: ...

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a metadata value by *key*, returning *default* if absent.

        Parameters
        ----------
        key : str
            Metadata key to look up.
        default : Any, optional
            Value to return when *key* is not present. Defaults to ``None``.

        Returns
        -------
        Any
            The metadata value, or *default*.

        """
        return self.metadata.get(key, default)

    def require(self, key: str) -> Any:  # noqa: ANN401
        """Look up a metadata value by *key*, raising if absent.

        Parameters
        ----------
        key : str
            Metadata key to look up.

        Returns
        -------
        Any
            The metadata value.

        Raises
        ------
        KeyError
            If *key* is not present in the metadata, with a message listing
            the available keys for easier debugging.

        """
        try:
            return self.metadata[key]
        except KeyError:
            available = ", ".join(sorted(self.metadata)) or "(none)"
            msg = (
                f"Required metadata key {key!r} not found in Explicit context. "
                f"Available keys: {available}."
            )
            raise KeyError(msg) from None

    def __contains__(self, key: str) -> bool:
        """Check whether *key* exists in the metadata."""
        return key in self.metadata


@dataclass(slots=True, frozen=True)
class Loaded:
    """Indicates a loaded map context; the map graph is directly provided."""

    map: MapGraph
    """The map graph directly provided for the scene."""
    tag: Literal["Loaded"] = "Loaded"


@dataclass(slots=True, frozen=True)
class NoMap:
    """Indicates no map is available for the scene."""

    tag: Literal["NoMap"] = "NoMap"


MapContext = Implicit | Explicit | Loaded | NoMap
