from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from dronalize.processing.filters.context import FilterContext


class Rule(ABC):
    """Base class shared by all filtering rules."""

    @abstractmethod
    def expr(self, ctx: FilterContext) -> pl.Expr:
        """Return the Polars expression that evaluates the rule."""


class FilterRule(Rule, ABC):
    """Base class for validation rules that decide scene validity."""


class AgentFilterRule(FilterRule, ABC):
    """Base class for per-agent validation rules."""

    max_invalid_agents: int
    max_invalid_fraction: float


class SceneFilterRule(FilterRule, ABC):
    """Base class for scene-level validation rules."""


class CleanupRule(Rule, ABC):
    """Base class for rules that physically remove rows before validation."""


def rule_name(rule: Rule) -> str:
    """Return the stable rule identifier used in config and diagnostics."""
    name = getattr(rule, "type", None)
    if not isinstance(name, str):
        msg = f"Rule {type(rule).__name__} is missing a string 'type' discriminator."
        raise TypeError(msg)
    return name
