from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import polars as pl
from pydantic import ConfigDict, Field, dataclasses, model_validator
from typing_extensions import override

from dronalize.core.models import AbsoluteTolerance, RelativeTolerance, Tolerance, tol
from dronalize.processing.filters.base import AgentCheckRule, RuleId
from dronalize.processing.filters.context import (
    AgentSelector,
    FrameInput,
    FrameSet,
    coerce_frame_set,
)

if TYPE_CHECKING:
    from dronalize.processing.filters.context import FilterContext


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class RequireFrames(AgentCheckRule):
    """Require each retained agent to cover specific relative frames."""

    frames: FrameSet
    type: Literal["frames"] = Field("frames", repr=False, init=False)

    @classmethod
    def define(
        cls,
        frames: FrameInput,
        *,
        selector: AgentSelector | None = None,
        tolerance: Tolerance | None = None,
        rule_id: RuleId | None = None,
    ) -> RequireFrames:
        """Return a defined rule."""
        return cls(
            frames=coerce_frame_set(frames),
            selector=selector,
            tolerance=(tolerance if tolerance is not None else tol(absolute=0)),
            rule_id=rule_id,
        )

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        relative_frame = ctx.relative_frame()
        required = relative_frame.filter(relative_frame.is_in(self.frames)).n_unique()
        return ctx.over_agent_window(required) == len(self.frames)


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class RequireWindow(AgentCheckRule):
    """Require a minimum fraction of frames in a relative agent window."""

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    type: Literal["window"] = Field("window", repr=False, init=False)

    @model_validator(mode="after")
    def _validate_window(self) -> RequireWindow:
        if self.end_frame < self.start_frame:
            msg = "end_frame must be greater than or equal to start_frame."
            raise ValueError(msg)
        return self

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        relative_frame = ctx.relative_frame()
        in_window = relative_frame.is_between(self.start_frame, self.end_frame, closed="both")
        covered_frames = ctx.over_agent_window(relative_frame.filter(in_window).n_unique())
        required_coverage = (self.end_frame - self.start_frame + 1) * self.min_fraction
        return covered_frames.cast(pl.Float64) >= required_coverage


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class MinSamples(AgentCheckRule):
    """Require a minimum number of samples per agent."""

    minimum: int = Field(ge=1)
    type: Literal["min_samples"] = Field("min_samples", repr=False, init=False)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        return ctx.over_agent_window(pl.len()) >= self.minimum


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class MaxMissingFrames(AgentCheckRule):
    """Require a maximum number of missing frames per agent."""

    maximum: int = Field(ge=0)
    type: Literal["max_missing_frames"] = Field("max_missing_frames", repr=False, init=False)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        scene_length = ctx.scene_length()
        agent_length = ctx.over_agent_window(pl.col(ctx.frame_column).n_unique())
        missing_frames = scene_length - agent_length
        return missing_frames <= self.maximum


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class MaxGap(AgentCheckRule):
    """Require the gap between two be less than or equal to a give maximum."""

    maximum: int = Field(ge=0)
    type: Literal["max_gap"] = Field("max_gap", repr=False, init=False)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        frame_col = pl.col(ctx.frame_column)
        gaps = frame_col - frame_col.shift(1) - 1
        max_gap = ctx.over_agent_window(gaps.max())
        return max_gap <= self.maximum


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class MinConsecutiveFrames(AgentCheckRule):
    """Require a minimum longest consecutive run per agent."""

    minimum: int = Field(ge=1)
    type: Literal["min_consecutive_frames"] = Field(
        "min_consecutive_frames", repr=False, init=False
    )

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        frame_col = pl.col(ctx.frame_column)
        prev_frame = frame_col.shift(1)
        frame_diff = frame_col - prev_frame
        new_run = prev_frame.is_null() | (frame_diff != 1)
        row_idx = ctx.over_agent_window(frame_col.cum_count())
        run_start_idx = ctx.over_agent_window(
            pl.when(new_run).then(row_idx).otherwise(None).forward_fill()
        )
        streak_len = row_idx - run_start_idx + 1
        longest_run = ctx.over_agent_window(streak_len.max())
        return longest_run >= self.minimum


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class StartsByFrame(AgentCheckRule):
    """Require each retained agent to start by a specific relative frame."""

    frame: int = Field(ge=0)
    type: Literal["starts_by_frame"] = Field("starts_by_frame", repr=False, init=False)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        first_frame = ctx.over_agent_window(pl.col(ctx.frame_column).min())
        return first_frame <= self.frame


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class EndsAfterFrame(AgentCheckRule):
    """Require each retained agent to end after a specific relative frame."""

    frame: int = Field(ge=0)
    type: Literal["ends_after_frame"] = Field("ends_after_frame", repr=False, init=False)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        last_frame = ctx.over_agent_window(pl.col(ctx.frame_column).max())
        return last_frame >= self.frame


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True), kw_only=True)
class MinSpan(AgentCheckRule):
    """Require a minimum span between the first and last frame of each retained agent."""

    minimum: int = Field(ge=1)
    type: Literal["min_span"] = Field("min_span", repr=False, init=False)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        first_frame = ctx.over_agent_window(pl.col(ctx.frame_column).min())
        last_frame = ctx.over_agent_window(pl.col(ctx.frame_column).max())
        span = last_frame - first_frame + 1
        return span >= self.minimum


def invalid_agent_tolerance_expr(
    tolerance: Tolerance, *, invalid_agents: pl.Expr, invalid_fraction: pl.Expr
) -> pl.Expr:
    """Return the scene-pass expression for a configured invalid-agent tolerance."""
    if isinstance(tolerance, RelativeTolerance):
        return invalid_fraction <= tolerance.relative
    if isinstance(tolerance, AbsoluteTolerance):
        return invalid_agents <= tolerance.absolute
    return pl.all_horizontal(
        invalid_agents <= tolerance.absolute, invalid_fraction <= tolerance.relative
    )


AgentCheckSpec = Annotated[
    MaxMissingFrames
    | RequireFrames
    | RequireWindow
    | MinSamples
    | MaxGap
    | MinConsecutiveFrames
    | StartsByFrame
    | EndsAfterFrame
    | MinSpan,
    Field(discriminator="type"),
]

__all__ = [
    "AgentCheckRule",
    "AgentCheckSpec",
    "EndsAfterFrame",
    "MaxGap",
    "MaxMissingFrames",
    "MinConsecutiveFrames",
    "MinSamples",
    "MinSpan",
    "RequireFrames",
    "RequireWindow",
    "StartsByFrame",
]
