"""Concrete node and map object implementations with integer IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from preprocessing.core.interface.map_protocols import BaseMapObject, BaseNode


@dataclass(init=False)
class IntIDBaseMapObject(BaseMapObject[int]):
    """Base class for map objects with an integer ID.

    This class is used as a base for map objects that have an integer ID.
    It provides a common interface for creating instances from dictionaries.
    """

    def __init__(self, object_id: int | None = None) -> None:
        """Initialize an `IntIDBaseMapObject` with a unique integer ID."""
        cls = type(self)
        if not hasattr(cls, "_id_counter"):
            cls._id_counter = 0  # Initialize counter for the subclass
        if object_id is None:
            self.id = cls._id_counter
            cls._id_counter += 1
        else:
            self.id = object_id

    @classmethod
    def reset_counter(cls) -> None:
        """Reset the ID counter to 0."""
        cls._id_counter = 0


# Backward compatibility alias
IntIdBaseMapObject = IntIDBaseMapObject


@dataclass(init=False)
class IntIDNode(IntIDBaseMapObject, BaseNode[int]):
    """A node in a 3D space with integer ID.

    The id is automatically assigned and unique for each instance, if not
    misused.
    """

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        """Initialize an `IntIDNode` with x, y, and optional z coordinates."""
        super().__init__()
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def with_id(cls, object_id: int, x: float, y: float, z: float = 0.0) -> IntIDNode:
        """Create an `IntIDNode` with a specified ID and coordinates."""
        node = cls(x, y, z)
        node.id = object_id
        return node

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntIDNode:
        """Create an `IntIDNode` instance from a dictionary."""
        return IntIDNode(
            x=data["x"],
            y=data["y"],
            z=data.get("z", 0.0),
        )
