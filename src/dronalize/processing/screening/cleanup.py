"""Cleanup rules that prune rows or agents before scene screening."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import polars as pl
from pydantic import Field
from typing_extensions import override

from dronalize.core.categories import AgentCategoryInput, coerce_agent_categories
from dronalize.processing.screening.agent import AgentCheckRule  # noqa: TC001
from dronalize.processing.screening.base import CleanupRuleBase, RuleId
from dronalize.processing.screening.context import AgentSet  # noqa: TC001

if TYPE_CHECKING:
    from dronalize.processing.screening.context import ScreeningContext


class PruneByRule(CleanupRuleBase):
    """Remove rows for agents that fail the given screening rule."""

    agent_rule: AgentCheckRule
    rule: Literal["prune_by"] = Field("prune_by", repr=False, init=False)

    @override
    def expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the row-retention expression for the wrapped agent rule."""
        scope = ctx.selector_mask(self.agent_rule.selector)
        agent_validity = self.agent_rule.expr(ctx)
        return pl.when(scope).then(agent_validity).otherwise(statement=True)


class ExcludeCategories(CleanupRuleBase):
    """Remove rows that belong to the given agent categories."""

    categories: AgentSet
    rule: Literal["exclude"] = Field("exclude", repr=False, init=False)

    @classmethod
    def define(
        cls, categories: AgentCategoryInput, *, rule_id: RuleId | None = None
    ) -> ExcludeCategories:
        """Alternate constructor that accepts one or many category values."""
        return cls(categories=coerce_agent_categories(categories, frozenset), rule_id=rule_id)

    @override
    def expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the row-retention expression that excludes selected categories."""
        return ~pl.col(ctx.category_column).is_in(self.categories)


class IncludeCategories(CleanupRuleBase):
    """Keep only rows that belong to the given agent categories."""

    categories: AgentSet
    rule: Literal["include"] = Field("include", repr=False, init=False)

    @classmethod
    def define(
        cls, categories: AgentCategoryInput, *, rule_id: RuleId | None = None
    ) -> IncludeCategories:
        """Alternate constructor that accepts one or many category values."""
        return cls(categories=coerce_agent_categories(categories, frozenset), rule_id=rule_id)

    @override
    def expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the row-retention expression that keeps selected categories."""
        return pl.col(ctx.category_column).is_in(self.categories)


CleanupRule = Annotated[
    PruneByRule | ExcludeCategories | IncludeCategories, Field(discriminator="rule")
]

__all__ = [
    "CleanupRule",
    "CleanupRuleBase",
    "ExcludeCategories",
    "IncludeCategories",
    "PruneByRule",
]
