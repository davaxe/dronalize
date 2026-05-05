"""Base rule models and protocols for scene-screening definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import override

from dronalize.config.models.screening import Tolerance  # noqa: TC001
from dronalize.processing.screening.context import AgentSelector

if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl

    from dronalize.processing.screening.context import ScreeningContext

RuleId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+$")]


class Rule(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]: https://docs.pydantic.dev/1.10/usage/models/?utm_source=chatgpt.com#abstract-base-classes
    """Base class shared by all screening rules."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(default=True, repr=False)
    rule_id: RuleId | None = Field(default=None, repr=False)

    @abstractmethod
    def expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the Polars expression that evaluates the rule."""

    def name(self) -> str:
        """Return the stable rule identifier used in diagnostics and merging."""
        return rule_name(self)


class CheckRuleBase(Rule, ABC):
    """Base class for rules that contribute pass/fail checks."""


class AgentCheckRuleBase(CheckRuleBase, ABC):
    """Base class for per-agent check rules."""

    selector: AgentSelector | None = None
    tolerance: Tolerance | None = Field(default=None)

    @override
    def __repr_args__(self) -> Iterator[tuple[str | None, object]]:
        """Omit selector from repr when none."""
        for name, value in super().__repr_args__():
            if name == "selector" and value is None:
                continue
            yield name, value


class SceneCheckRuleBase(CheckRuleBase, ABC):
    """Base class for scene-level check rules."""


class CleanupRuleBase(Rule, ABC):
    """Base class for rules that physically remove rows before screening."""


def rule_group(rule: Rule) -> str:
    """Return the rule group used to namespace implicit rule identifiers."""
    if isinstance(rule, AgentCheckRuleBase):
        return "agent"
    if isinstance(rule, SceneCheckRuleBase):
        return "scene"
    if isinstance(rule, CleanupRuleBase):
        return "cleanup"
    return "rule"


def rule_name(rule: Rule) -> str:
    """Return the stable rule identifier used in diagnostics and merging."""
    if rule.rule_id is not None:
        return rule.rule_id
    name = getattr(rule, "rule", None)
    if not isinstance(name, str):
        msg = f"Rule {type(rule).__name__} is missing a string 'rule' discriminator."
        raise TypeError(msg)
    return f"{rule_group(rule)}_{name}"


_ = AgentSelector
