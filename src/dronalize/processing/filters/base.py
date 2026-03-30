from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated

from pydantic import ConfigDict, Field, StringConstraints, dataclasses

from dronalize.core.models import Tolerance, tol
from dronalize.processing.filters.context import AgentSelector

if TYPE_CHECKING:
    import polars as pl

    from dronalize.processing.filters.context import FilterContext

RuleId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+$")]


@dataclasses.dataclass(config=ConfigDict(frozen=True), kw_only=True)
class Rule(ABC):
    """Base class shared by all filtering rules."""

    rule_id: RuleId | None = None

    @abstractmethod
    def expr(self, ctx: FilterContext) -> pl.Expr:
        """Return the Polars expression that evaluates the rule."""

    def name(self) -> str:
        """Return the stable rule identifier used in diagnostics and merging."""
        return rule_name(self)


@dataclasses.dataclass(config=ConfigDict(frozen=True), kw_only=True)
class CheckRule(Rule, ABC):
    """Base class for rules that contribute pass/fail checks."""


@dataclasses.dataclass(config=ConfigDict(frozen=True), kw_only=True)
class AgentCheckRule(CheckRule, ABC):
    """Base class for per-agent check rules."""

    selector: AgentSelector | None = None
    tolerance: Tolerance = Field(default=tol(absolute=0))


@dataclasses.dataclass(config=ConfigDict(frozen=True), kw_only=True)
class SceneCheckRule(CheckRule, ABC):
    """Base class for scene-level check rules."""


@dataclasses.dataclass(config=ConfigDict(frozen=True), kw_only=True)
class CleanupRule(Rule, ABC):
    """Base class for rules that physically remove rows before validation."""


def rule_group(rule: Rule) -> str:
    """Return the rule group used to namespace implicit rule identifiers."""
    if isinstance(rule, AgentCheckRule):
        return "agent"
    if isinstance(rule, SceneCheckRule):
        return "scene"
    if isinstance(rule, CleanupRule):
        return "cleanup"
    return "rule"


def rule_name(rule: Rule) -> str:
    """Return the stable rule identifier used in diagnostics and merging."""
    if rule.rule_id is not None:
        return rule.rule_id
    name = getattr(rule, "type", None)
    if not isinstance(name, str):
        msg = f"Rule {type(rule).__name__} is missing a string 'type' discriminator."
        raise TypeError(msg)
    return f"{rule_group(rule)}_{name}"


_ = AgentSelector
