from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import polars as pl
from pydantic import ConfigDict, Field, dataclasses
from typing_extensions import override

from dronalize.core.categories import AgentCategoryInput, coerce_agent_categories
from dronalize.processing.filters.base import AgentCheckRule, CleanupRule, RuleId
from dronalize.processing.filters.context import AgentSet  # noqa: TC001

if TYPE_CHECKING:
    from dronalize.processing.filters.context import FilterContext


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class PruneByRule(CleanupRule):
    """Remove rows for agents that fail the given validation rule."""

    rule: AgentCheckRule
    type: Literal["prune_by_rule"] = Field("prune_by_rule", repr=False, init=False)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        scope = ctx.selector_mask(self.rule.selector)
        agent_validity = self.rule.expr(ctx)
        return pl.when(scope).then(agent_validity).otherwise(statement=True)


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class ExcludeCategories(CleanupRule):
    """Remove rows that belong to the given agent categories."""

    categories: AgentSet
    type: Literal["exclude"] = Field("exclude", repr=False, init=False)

    @classmethod
    def define(
        cls, categories: AgentCategoryInput, *, rule_id: RuleId | None = None
    ) -> ExcludeCategories:
        """Alternate constructor that accepts one or many category values."""
        return cls(categories=coerce_agent_categories(categories, frozenset), rule_id=rule_id)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        return ~pl.col(ctx.category_column).is_in(self.categories)


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class IncludeCategories(CleanupRule):
    """Keep only rows that belong to the given agent categories."""

    categories: AgentSet
    type: Literal["include"] = Field("include", repr=False, init=False)

    @classmethod
    def define(
        cls, categories: AgentCategoryInput, *, rule_id: RuleId | None = None
    ) -> IncludeCategories:
        """Alternate constructor that accepts one or many category values."""
        return cls(categories=coerce_agent_categories(categories, frozenset), rule_id=rule_id)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        return pl.col(ctx.category_column).is_in(self.categories)


CleanupSpec = Annotated[ExcludeCategories | IncludeCategories, Field(discriminator="type")]

__all__ = ["CleanupRule", "CleanupSpec", "ExcludeCategories", "IncludeCategories", "PruneByRule"]
