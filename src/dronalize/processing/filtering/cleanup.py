"""Cleanup rules that prune rows or agents before scene validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import polars as pl
from pydantic import Field
from typing_extensions import override

from dronalize.core.categories import AgentCategoryInput, coerce_agent_categories
from dronalize.processing.filtering.agent import AgentCheckSpec  # noqa: TC001
from dronalize.processing.filtering.base import CleanupRule, RuleId
from dronalize.processing.filtering.context import AgentSet  # noqa: TC001

if TYPE_CHECKING:
    from dronalize.processing.filtering.context import FilterContext


class PruneByRule(CleanupRule):
    """Remove rows for agents that fail the given validation rule."""

    rule: AgentCheckSpec
    type: Literal["prune_by_rule"] = Field("prune_by_rule", repr=False, init=False)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        """Return the row-retention expression for the wrapped agent rule."""
        scope = ctx.selector_mask(self.rule.selector)
        agent_validity = self.rule.expr(ctx)
        return pl.when(scope).then(agent_validity).otherwise(statement=True)


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
        """Return the row-retention expression that excludes selected categories."""
        return ~pl.col(ctx.category_column).is_in(self.categories)


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
        """Return the row-retention expression that keeps selected categories."""
        return pl.col(ctx.category_column).is_in(self.categories)


CleanupSpec = Annotated[
    PruneByRule | ExcludeCategories | IncludeCategories,
    Field(discriminator="type"),
]

__all__ = ["CleanupRule", "CleanupSpec", "ExcludeCategories", "IncludeCategories", "PruneByRule"]

_ = PruneByRule.model_rebuild(_types_namespace=globals())
