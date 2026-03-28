from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import polars as pl
from pydantic import BeforeValidator, Field
from pydantic.dataclasses import dataclass
from typing_extensions import override

from dronalize.processing.filters.rules.base import AgentFilterRule

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.processing.filters.context import FilterContext


def coerce_frame_set(value: int | Iterable[int]) -> FrameSet:
    """Convert one or many frame indices into a normalized integer set."""
    values = [value] if isinstance(value, int) else value
    return frozenset(int(item) for item in values)


FrameSet = Annotated[frozenset[int], Field(min_length=1), BeforeValidator(coerce_frame_set)]


@dataclass(slots=True, frozen=True)
class RequireFullAgentWindow(AgentFilterRule):
    """Require each retained agent to span the full scene window."""

    max_invalid_agents: int = Field(default=0, ge=0)
    max_invalid_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    type: Literal["require_full_agent_window"] = Field(
        "require_full_agent_window", repr=False, init=False
    )

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        scene_length = ctx.over_scene_window(pl.col(ctx.frame_column).n_unique())
        agent_length = ctx.over_agent_window(pl.col(ctx.frame_column).n_unique())
        return agent_length == scene_length


@dataclass(slots=True, frozen=True)
class RequireAgentFrames(AgentFilterRule):
    """Require each retained agent to cover specific relative frames."""

    frames: FrameSet
    max_invalid_agents: int = Field(default=0, ge=0)
    max_invalid_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    type: Literal["require_agent_frames"] = Field("require_agent_frames", repr=False, init=False)

    @classmethod
    def define(
        cls,
        frames: Iterable[int],
        max_invalid_agents: int = 0,
        max_invalid_fraction: float = 0.0,
    ) -> RequireAgentFrames:
        """Return a defined rule.

        This constructor allow for mor flexible input in comparison to the
        default dataclass constructor.
        """
        return cls(
            frames=frozenset(frames),
            max_invalid_agents=max_invalid_agents,
            max_invalid_fraction=max_invalid_fraction,
        )

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        relative_frame = ctx.relative_frame()
        required = relative_frame.filter(relative_frame.is_in(self.frames)).n_unique()
        return ctx.over_agent_window(required) == len(self.frames)


@dataclass(slots=True, frozen=True)
class MinimumAgentSamples(AgentFilterRule):
    """Require a minimum number of samples per retained agent."""

    minimum: int = Field(ge=1)
    max_invalid_agents: int = Field(default=0, ge=0)
    max_invalid_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    type: Literal["minimum_agent_samples"] = Field("minimum_agent_samples", repr=False, init=False)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        return ctx.over_agent_window(pl.len()) >= self.minimum
