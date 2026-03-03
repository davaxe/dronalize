"""Concrete node and map object implementations with integer IDs."""

from __future__ import annotations

from dataclasses import dataclass

from dronalize.core.protocols.map_object import BaseNode


@dataclass(init=False)
class IntIDBaseMapObject:
    """Base class for map objects with an integer ID.

    This class is used as a base for map objects that have an integer ID.
    """

    def __init__(self, object_id: int) -> None:
        """Initialize an `IntIDBaseMapObject` with the given integer ID.

        Parameters
        ----------
        object_id : int
            The unique integer ID for this map object.

        """
        self.id = object_id


@dataclass(init=False)
class IntIDNode(IntIDBaseMapObject, BaseNode[int]):
    """A node in a 3D space with integer ID.

    IDs are assigned externally — typically by a `GraphBuilder` which owns the
    counter and guarantees uniqueness within a single graph-building session.
    """

    def __init__(self, object_id: int, x: float, y: float, z: float = 0.0) -> None:
        """Initialize an `IntIDNode` with an explicit ID and coordinates.

        Parameters
        ----------
        object_id : int
            The unique integer ID for this node.
        x : float
            X coordinate.
        y : float
            Y coordinate.
        z : float, optional
            Z coordinate. Defaults to 0.0.

        """
        super().__init__(object_id)
        self.x = x
        self.y = y
        self.z = z
