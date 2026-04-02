"""Enumerations and coercion helpers for shared categorical types."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from enum import Enum, IntEnum
from typing import TypeVar


class AgentCategory(IntEnum):
    """Enumeration of categories of agents / objects.

    !!! note "Dataset-dependent categories"
        Agent categories are not standardized across datasets, and are mapped
        from their original dataset-specific labels into this shared set of
        categories as closely as possible.

    """

    ANIMAL = 1
    """Non-human agents, including pets and wildlife."""
    BICYCLE = 2
    """Two-wheeled non-motorized vehicles."""
    BUS = 3
    """Large passenger vehicles, including city buses."""
    CAR = 4
    """Passenger vehicles, including sedans, SUVs, and pickup trucks."""
    EMERGENCY_VEHICLE = 5
    """Emergency response vehicles."""
    MOTORCYCLE = 6
    """Two-wheeled motorized vehicles."""
    MOVEABLE_OBJECT = 7
    """Generic category for moveable-objects (dataset dependent)."""
    PEDESTRIAN = 8
    """People walking or running."""
    STATIC_OBJECT = 9
    """Generic category for static-objects (dataset dependent)."""
    TRAILER = 10
    """Semi-trailers and other trailer types."""
    TRAM = 11
    """Street-level rail vehicles."""
    TRICYCLE = 12
    """Three-wheeled vehicles."""
    TRUCK = 13
    """Heavy-duty vehicles."""
    UNIMPORTANT = 14
    """Agents that are unimportant to model for a given dataset or task."""
    UNKNOWN = 15
    """Agents of unknown or unclassifiable category."""
    VAN = 16
    """Larger passenger vehicles, including minivans and cargo vans."""

    @classmethod
    def from_string(cls, value: str) -> AgentCategory:
        """Convert a string to an AgentCategory, case-insensitive."""
        s: str = value.lower()
        for category in AgentCategory:
            if category.name.lower() == s:
                return category
        msg = f"Unknown agent category: {s}"
        raise ValueError(msg)

    @classmethod
    def from_value(cls, value: int | str | AgentCategory) -> AgentCategory:
        """Convert an integer or string to an AgentCategory."""
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls.from_string(value)


class DatasetSplit(str, Enum):
    """Enum representing the available dataset splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


AgentCategoryLike = int | str | AgentCategory
AgentCategoryInput = AgentCategoryLike | Iterable[AgentCategoryLike]

T = TypeVar("T", bound=Collection[AgentCategory])


def coerce_agent_categories(
    value: AgentCategoryInput, collection: Callable[[Iterable[AgentCategory]], T]
) -> T:
    """Convert one or many agent-category values into a normalized collection."""
    values = [value] if isinstance(value, (str, int, AgentCategory)) else value
    return collection(AgentCategory.from_value(item) for item in values)
