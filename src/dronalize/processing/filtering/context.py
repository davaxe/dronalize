"""Shared expression-building context for scene and agent filters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from dronalize.core.categories import AgentCategory, AgentCategoryInput, coerce_agent_categories


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


class AgentSelector(BaseModel):
    """Restrict a rule to agents matching category predicates."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    mode: Literal["include", "exclude"]
    categories: AgentSet

    @classmethod
    def define(
        cls, mode: Literal["include", "exclude"], categories: AgentCategoryInput
    ) -> AgentSelector:
        """Create a selector from flexible category inputs."""
        return cls(mode=mode, categories=coerce_agent_categories(categories, frozenset))

    @classmethod
    def include(cls, categories: AgentCategoryInput) -> AgentSelector:
        """Create a selector that keeps only the given categories in scope."""
        return cls.define("include", categories)

    @classmethod
    def exclude(cls, categories: AgentCategoryInput) -> AgentSelector:
        """Create a selector that excludes the given categories from scope."""
        return cls.define("exclude", categories)


@dataclass(slots=True, frozen=True)
class FilterContext:
    """Shared expressions and column metadata used while evaluating filter logic."""

    agent_id: str
    frame_column: str
    category_column: str
    scene_window: pl.Expr | list[str]
    agent_window: list[str]

    def over_scene_window(self, expr: pl.Expr) -> pl.Expr:
        """Apply the scene window to an expression."""
        if isinstance(self.scene_window, list) and not self.scene_window:
            return expr
        return expr.over(self.scene_window)

    def over_agent_window(self, expr: pl.Expr) -> pl.Expr:
        """Apply the agent window to an expression."""
        if not self.agent_window:
            return expr
        return expr.over(self.agent_window)

    def relative_frame(self) -> pl.Expr:
        """Return the scene-relative frame index expression."""
        return pl.col(self.frame_column) - self.scene_start_frame()

    def scene_length(self) -> pl.Expr:
        """Return the number of frames in the current scene."""
        return self.scene_end_frame() - self.scene_start_frame() + 1

    def scene_start_frame(self) -> pl.Expr:
        """Return the starting frame of the current scene."""
        return self.over_scene_window(pl.col(self.frame_column).min())

    def scene_end_frame(self) -> pl.Expr:
        """Return the ending frame of the current scene."""
        return self.over_scene_window(pl.col(self.frame_column).max())

    def selector_mask(self, selector: AgentSelector | None) -> pl.Expr:
        """Return a row mask for the given selector."""
        if selector is None:
            return pl.lit(value=True)

        in_scope = pl.col(self.category_column).is_in(selector.categories)
        return in_scope if selector.mode == "include" else ~in_scope

    def retained_agent_count(self, selector: AgentSelector | None = None) -> pl.Expr:
        """Return the number of retained agents in the current scene."""
        return self.over_scene_window(
            pl.col(self.agent_id).filter(self.selector_mask(selector)).n_unique()
        )
