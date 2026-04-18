"""Scene-scoped screening rules for trajectory scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import polars as pl
from pydantic import Field, model_validator
from typing_extensions import override

from dronalize.config.models import Range  # noqa: TC001
from dronalize.core.categories import AgentCategory, AgentCategoryInput
from dronalize.processing.screening.base import RuleId, SceneCheckRuleBase
from dronalize.processing.screening.context import (
    AgentSelector,
    FrameInput,
    FrameSet,
    coerce_frame_set,
)

if TYPE_CHECKING:
    from dronalize.processing.screening.context import ScreeningContext


class AgentRange(SceneCheckRuleBase):
    """Require a minimum and/or maximum number of retained agents in the scene."""

    minimum: int | None = Field(default=None, ge=0)
    maximum: int | None = Field(default=None, ge=0)
    selector: AgentSelector | None = None
    rule: Literal["agent_range"] = Field("agent_range", repr=False, init=False)

    @model_validator(mode="after")
    def _val_range(self) -> AgentRange:
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            msg = "maximum must be greater than or equal to minimum."
            raise ValueError(msg)
        return self

    @override
    def expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the scene-pass expression for the retained-agent count range."""
        count = ctx.retained_agent_count(self.selector)
        cond = pl.lit(value=True)
        if self.minimum is not None:
            cond &= count >= self.minimum
        if self.maximum is not None:
            cond &= count <= self.maximum
        return cond


class CategoryRange(SceneCheckRuleBase):
    """Require a minimum and/or maximum number of retained agents in each category."""

    ranges: dict[AgentCategory, Range]
    rule: Literal["category_range"] = Field("category_range", repr=False, init=False)

    @classmethod
    def define(cls, *ranges: tuple[AgentCategoryInput, Range]) -> CategoryRange:
        """Alternate constructor that accepts one or many category-range pairs."""
        return cls(ranges={AgentCategory(category): r for category, r in ranges})

    @override
    def expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the scene-pass expression for the per-category agent ranges."""
        cond = pl.lit(value=True)
        for category, r in self.ranges.items():
            count = ctx.retained_agent_count(AgentSelector.include(category))
            if r.minimum is not None:
                cond &= count >= r.minimum
            if r.maximum is not None:
                cond &= count <= r.maximum
        return cond


class RequireFrames(SceneCheckRuleBase):
    """Require specific relative frames to exist in the scene window."""

    frames: FrameSet
    rule: Literal["frames"] = Field("frames", repr=False, init=False)

    @classmethod
    def define(cls, frames: FrameInput, *, rule_id: RuleId | None = None) -> RequireFrames:
        """Alternate constructor that accepts one or many frame indices."""
        return cls(frames=coerce_frame_set(frames), rule_id=rule_id)

    @override
    def expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the scene-pass expression for the required frame set."""
        relative_frame = ctx.relative_frame()
        required = relative_frame.filter(relative_frame.is_in(self.frames)).n_unique()
        return ctx.over_scene_window(required) == len(self.frames)


class RequireWindow(SceneCheckRuleBase):
    """Require a minimum fraction of frames in a relative scene window."""

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    rule: Literal["window"] = Field("window", repr=False, init=False)

    @model_validator(mode="after")
    def _validate_window(self) -> RequireWindow:
        if self.end_frame < self.start_frame:
            msg = "end_frame must be greater than or equal to start_frame."
            raise ValueError(msg)
        return self

    @override
    def expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the scene-pass expression for the required window coverage."""
        relative_frame = ctx.relative_frame()
        in_window = relative_frame.is_between(self.start_frame, self.end_frame, closed="both")
        covered_frames = ctx.over_scene_window(relative_frame.filter(in_window).n_unique())
        required_coverage = round((self.end_frame - self.start_frame + 1) * self.min_fraction)
        return covered_frames >= required_coverage


class MaxMissingFrames(SceneCheckRuleBase):
    """Require the cleaned scene window to stay within a missing-frame budget."""

    max_missing_frames: int = Field(default=0, ge=0)
    rule: Literal["max_missing_frames"] = Field("max_missing_frames", repr=False, init=False)

    @override
    def expr(self, ctx: ScreeningContext) -> pl.Expr:
        """Return the scene-pass expression for the missing-frame budget."""
        unique_frame_count = ctx.over_scene_window(pl.col(ctx.columns.frame).n_unique())
        frame_span = ctx.over_scene_window(
            pl.col(ctx.columns.frame).max() - pl.col(ctx.columns.frame).min() + 1
        )
        return (frame_span - unique_frame_count) <= self.max_missing_frames


SceneCheckRule = Annotated[
    AgentRange | CategoryRange | RequireFrames | RequireWindow | MaxMissingFrames,
    Field(discriminator="rule"),
]
"""Discriminated union of executable scene-scoped screening rule types.

These rule objects evaluate aggregate properties of a cleaned scene such as
frame coverage, retained-agent counts, or per-category count ranges.
"""

__all__ = [
    "AgentRange",
    "CategoryRange",
    "MaxMissingFrames",
    "RequireFrames",
    "RequireWindow",
    "SceneCheckRule",
    "SceneCheckRuleBase",
]
