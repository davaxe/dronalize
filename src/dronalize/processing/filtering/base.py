"""Base rule models and protocols for scene-filter definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import override

from dronalize.processing.filtering.context import AgentSelector
from dronalize.processing.filtering.tolerance import Tolerance, tol

if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl

    from dronalize.processing.filtering.context import FilterContext

RuleId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+$")]


class Rule(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]: https://docs.pydantic.dev/1.10/usage/models/?utm_source=chatgpt.com#abstract-base-classes
    """Base class shared by all filtering rules."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    rule_id: RuleId | None = Field(default=None, repr=False)

    @abstractmethod
    def expr(self, ctx: FilterContext) -> pl.Expr:
        """Return the Polars expression that evaluates the rule."""

    def name(self) -> str:
        """Return the stable rule identifier used in diagnostics and merging."""
        return rule_name(self)


class CheckRule(Rule, ABC):
    """Base class for rules that contribute pass/fail checks."""


class AgentCheckRule(CheckRule, ABC):
    """Base class for per-agent check rules."""

    selector: AgentSelector | None = None
    tolerance: Tolerance = Field(default=tol(absolute=0))

    @override
    def __repr_args__(self) -> Iterator[tuple[str | None, object]]:
        """Omit selector from repr when none."""
        for name, value in super().__repr_args__():
            if name == "selector" and value is None:
                continue
            yield name, value


class SceneCheckRule(CheckRule, ABC):
    """Base class for scene-level check rules."""


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
