from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import Field

if TYPE_CHECKING:
    import polars as pl

    from dronalize.processing.filters.context import FilterContext


class Rule(ABC):
    """Base class shared by all filtering rules."""

    @abstractmethod
    def expr(self, ctx: FilterContext) -> pl.Expr:
        """Return the Polars expression that evaluates the rule."""

    def name(self) -> str:
        """Return the stable rule identifier used in config and diagnostics."""
        return rule_name(self)


class ValidationRule(Rule, ABC):
    """Base class for validation rules that decide scene validity."""


class AgentValidationRule(ValidationRule, ABC):
    """Base class for per-agent validation rules."""

    max_invalid_agents: int = Field(ge=0, kw_only=True)
    max_invalid_fraction: float = Field(ge=0.0, le=1.0, kw_only=True)


class SceneValidationRule(ValidationRule, ABC):
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
