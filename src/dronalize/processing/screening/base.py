"""Base context, selectors, and rule models for scene-screening definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

import polars as pl
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints
from typing_extensions import override

from dronalize.config.models.screening import Tolerance  # noqa: TC001
from dronalize.core.categories import AgentCategory, AgentCategoryInput, coerce_agent_categories

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dronalize.processing.columns import TrajectoryColumns


def coerce_frame_set(value: FrameInput) -> frozenset[int]:
    """Convert one or many frame indices into a normalized integer set."""
    values = [value] if isinstance(value, int) else value
    return frozenset(int(item) for item in values)


FrameInput = int | Iterable[int]
"""One or more frame indices relative to the start of the scene."""
FrameSet = Annotated[frozenset[int], Field(min_length=1), BeforeValidator(coerce_frame_set)]
"""Set of frame indices relative to the start of the scene."""
AgentSet = Annotated[
    frozenset[AgentCategory],
    Field(min_length=1),
    BeforeValidator(lambda v: coerce_agent_categories(v, frozenset)),
]
"""Set of agent categories."""


class AgentCategorySelector(BaseModel):
    """Restrict a rule to agents matching category predicates."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    mode: Literal["include", "exclude"]
    categories: AgentSet

    @classmethod
    def define(
        cls, mode: Literal["include", "exclude"], categories: AgentCategoryInput
    ) -> AgentCategorySelector:
        """Create a selector from flexible category inputs."""
        return cls(mode=mode, categories=coerce_agent_categories(categories, frozenset))

    @classmethod
    def include(cls, categories: AgentCategoryInput) -> AgentCategorySelector:
        """Create a selector that keeps only the given categories in scope."""
        return cls.define("include", categories)

    @classmethod
    def exclude(cls, categories: AgentCategoryInput) -> AgentCategorySelector:
        """Create a selector that excludes the given categories from scope."""
        return cls.define("exclude", categories)


@dataclass(slots=True, frozen=True)
class ScreeningContext:
    """Shared expressions and column metadata used while evaluating screening logic."""

    columns: TrajectoryColumns
    scene_window: tuple[str, ...]
    agent_window: tuple[str, ...]

    def over_scene_window(self, expr: pl.Expr) -> pl.Expr:
        """Apply the scene window to an expression."""
        if not self.scene_window:
            return expr
        return expr.over(self.scene_window)

    def over_agent_window(self, expr: pl.Expr) -> pl.Expr:
        """Apply the agent window to an expression."""
        if not self.agent_window:
            return expr
        return expr.over(self.agent_window)

    def relative_frame(self) -> pl.Expr:
        """Return the scene-relative frame index expression."""
        return pl.col(self.columns.frame) - self.scene_start_frame()

    def scene_length(self) -> pl.Expr:
        """Return the number of frames in the current scene."""
        return self.scene_end_frame() - self.scene_start_frame() + 1

    def scene_start_frame(self) -> pl.Expr:
        """Return the starting frame of the current scene."""
        return self.over_scene_window(pl.col(self.columns.frame).min())

    def scene_end_frame(self) -> pl.Expr:
        """Return the ending frame of the current scene."""
        return self.over_scene_window(pl.col(self.columns.frame).max())

    def selector_mask(self, selector: AgentCategorySelector | None) -> pl.Expr:
        """Return a row mask for the given selector."""
        if selector is None:
            return pl.lit(value=True)

        in_scope = pl.col(self.columns.category).is_in(selector.categories)
        return in_scope if selector.mode == "include" else ~in_scope

    def retained_agent_count(self, selector: AgentCategorySelector | None = None) -> pl.Expr:
        """Return the number of retained agents in the current scene."""
        return self.over_scene_window(
            pl.col(self.columns.agent_id).filter(self.selector_mask(selector)).n_unique()
        )


RuleId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+$")]


class Rule(BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]: https://docs.pydantic.dev/1.10/usage/models/?utm_source=chatgpt.com#abstract-base-classes
    """Base class shared by all screening rules."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(default=True, repr=False)
    rule_id: RuleId | None = Field(default=None, repr=False)

    @abstractmethod
    def predicate_expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the Polars expression that evaluates the rule."""

    def name(self) -> str:
        """Return the stable rule identifier used in diagnostics and merging."""
        return rule_name(self)


class CheckRuleBase(Rule, ABC):
    """Base class for rules that contribute pass/fail checks."""


class AgentCheckRuleBase(CheckRuleBase, ABC):
    """Base class for per-agent check rules."""

    selector: AgentCategorySelector | None = None
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
