from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from enum import Enum, IntEnum, auto
from typing import TypeVar


class AgentCategory(IntEnum):
    """Enumeration of categories of agents / objects."""

    CAR = auto()
    VAN = auto()
    TRAILER = auto()
    TRUCK = auto()
    TRAM = auto()
    BUS = auto()
    MOTORCYCLE = auto()
    BICYCLE = auto()
    PEDESTRIAN = auto()
    TRICYCLE = auto()
    ANIMAL = auto()
    STATIC_OBJECT = auto()
    MOVEABLE_OBJECT = auto()
    EMERGENCY_VEHICLE = auto()
    UNKNOWN = auto()
    UNIMPORTANT = auto()

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
