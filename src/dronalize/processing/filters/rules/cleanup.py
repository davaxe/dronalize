from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Annotated, Literal

import polars as pl
from pydantic import BeforeValidator, Field, dataclasses
from typing_extensions import override

from dronalize.core.categories import AgentCategory
from dronalize.processing.filters.rules.base import CleanupRule

if TYPE_CHECKING:
    from dronalize.processing.filters.context import FilterContext

AgentValue = int | str | AgentCategory
AgentInput = AgentValue | Iterable[AgentValue]
FrameInput = int | Iterable[int]


def _coerce_agent_set(value: AgentInput) -> AgentSet:
    """Convert one or many agent-category values into a normalized set."""
    values = [value] if isinstance(value, (str, int, AgentCategory)) else value
    return frozenset(AgentCategory.from_value(item) for item in values)


AgentSet = Annotated[
    frozenset[AgentCategory], Field(min_length=1), BeforeValidator(_coerce_agent_set)
]


@dataclasses.dataclass(slots=True, frozen=True)
class ExcludeAgentCategories(CleanupRule):
    """Remove rows that belong to the given agent categories."""

    categories: AgentSet
    type: Literal["exclude_categories"] = Field(
        "exclude_categories",
        repr=False,
        init=False,
    )

    @classmethod
    def define(cls, categories: AgentInput) -> ExcludeAgentCategories:
        """Alternate constructor that accepts one or many category values."""
        return cls(categories=_coerce_agent_set(categories))

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        category_column = ctx.category_column_or_raise()
        return ~pl.col(category_column).is_in(self.categories)


@dataclasses.dataclass(slots=True, frozen=True)
class IncludeAgentCategories(CleanupRule):
    """Keep only rows that belong to the given agent categories."""

    categories: AgentSet
    type: Literal["include_categories"] = Field(
        "include_categories",
        repr=False,
        init=False,
    )

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        category_column = ctx.category_column_or_raise()
        return pl.col(category_column).is_in(self.categories)


CleanupSpec = Annotated[
    ExcludeAgentCategories | IncludeAgentCategories, Field(discriminator="type")
]
