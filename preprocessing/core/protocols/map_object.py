"""Protocol and base class definitions for map objects and nodes."""

from __future__ import annotations

import math
from collections.abc import Hashable
from enum import IntEnum
from typing import Any, Generic, Protocol, TypeVar

from typing_extensions import Self

ID = TypeVar("ID", bound=Hashable)


class BaseMapObject(Protocol, Generic[ID]):
    """Base class for all map objects in a NuScenes map."""

    id: ID

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, Any]) -> Self:
        """Create an instance of the class from a dictionary."""
        ...

    @classmethod
    def try_from_dict(
        cls: type[Self],
        data: dict[str, Any],
    ) -> Self | None:
        """Try to create an instance from a dictionary, returning None on failure."""
        try:
            return cls.from_dict(data)
        except (ValueError, TypeError):
            return None


class BaseNode(Protocol, Generic[ID]):
    """Protocol for a node in a map.

    This protocol defines the interface for a node in a map, which includes
    methods for calculating distances to other nodes.
    """

    id: ID
    x: float
    y: float
    z: float = 0.0

    def distance_to(self, other: Self) -> float:
        """Calculate the Euclidean distance to another node.

        Parameters
        ----------
        other : Self
            The other node to calculate the distance to.

        Returns
        -------
        float
            Euclidean distance between this node and `other`.

        """
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2,
        )

    def distance_sq_to(self, other: Self) -> float:
        """Calculate the squared Euclidean distance to another node.

        Parameters
        ----------
        other : Self
            The other node to calculate the squared distance to.

        Returns
        -------
        float
            Squared Euclidean distance between this node and `other`.

        """
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2


class BaseEnum(IntEnum):
    """Base class for enums that can be created from strings.

    .. note::
        The naming of the enum members should match the expected string
        values, e.g. if the string representation is "EXAMPLE", the enum
        member should be named `EXAMPLE`. This is to ensure that the
        `from_str` and `try_from_str` methods work correctly.
    """

    @classmethod
    def from_str(cls: type[Self], value: str) -> Self:
        """Convert a string to an enum member, raising `ValueError` if not found.

        Parameters
        ----------
        value : str
            String representation of the enum member.

        Returns
        -------
        Self
            The matching enum member.

        Raises
        ------
        ValueError
            If `value` does not match any enum member.

        """
        segment = cls.try_from_str(value)

        if segment is not None:
            return segment

        msg: str = (
            f"Enum value '{value}' is not recognized. "
            f"Available values: {', '.join(cls.__members__.keys())}."
        )
        raise ValueError(msg)

    @classmethod
    def try_from_str(cls: type[Self], value: str | None) -> Self | None:
        """Try to convert a string to an enum member, returning `None` otherwise.

        Parameters
        ----------
        value : str or None
            String representation of the enum member.

        Returns
        -------
        Self or None
            The matching enum member, or `None` if not found.

        """
        if value is None:
            return None
        return cls.__members__.get(value, None)

    @classmethod
    def from_int(cls: type[Self], value: int) -> Self:
        """Convert an integer to an enum member.

        Parameters
        ----------
        value : int
            Integer value of the enum member.

        Returns
        -------
        Self
            The matching enum member.

        Raises
        ------
        ValueError
            If `value` does not match any enum member.

        """
        segment = cls.try_from_int(value)

        if segment is not None:
            return segment

        msg: str = (
            f"Enum value '{value}' is not recognized. "
            f"Available values: "
            f"{', '.join(map(str, cls._value2member_map_.keys()))}."
        )
        raise ValueError(msg)

    @classmethod
    def try_from_int(cls: type[Self], value: int | None) -> Self | None:
        """Try to convert a value to an enum member, returning `None` otherwise.

        Parameters
        ----------
        value : int or None
            Integer value of the enum member.

        Returns
        -------
        Self or None
            The matching enum member, or `None` if not found.

        """
        if value is None:
            return None
        return cls(value) if value in cls._value2member_map_ else None
