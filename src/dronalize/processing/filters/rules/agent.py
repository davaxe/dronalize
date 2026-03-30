from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

import polars as pl
from pydantic import BeforeValidator, ConfigDict, Field, dataclasses
from typing_extensions import override

from dronalize.processing.filters.rules.base import AgentValidationRule

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.processing.filters.context import FilterContext


def coerce_frame_set(value: int | Iterable[int]) -> FrameSet:
    """Convert one or many frame indices into a normalized integer set."""
    values = [value] if isinstance(value, int) else value
    return frozenset(int(item) for item in values)


FrameSet = Annotated[frozenset[int], Field(min_length=1), BeforeValidator(coerce_frame_set)]


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True))
class RequireCompleteAgentCoverage(AgentValidationRule):
    """Require each retained agent to span the full scene window."""

    max_invalid_agents: int = Field(default=0, ge=0)
    max_invalid_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    type: Literal["complete_agent_coverage"] = Field(
        "complete_agent_coverage", repr=False, init=False
    )

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        scene_length = ctx.over_scene_window(pl.col(ctx.frame_column).n_unique())
        agent_length = ctx.over_agent_window(pl.col(ctx.frame_column).n_unique())
        return agent_length == scene_length


@dataclasses.dataclass(slots=True, config=ConfigDict(frozen=True))
class RequireAgentCoverageAtFrames(AgentValidationRule):
    """Require each retained agent to cover specific relative frames."""

    frames: FrameSet
    max_invalid_agents: int = Field(default=0, ge=0)
    max_invalid_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    type: Literal["agent_frames"] = Field(
        "agent_frames",
        repr=False,
        init=False,
    )

    @classmethod
    def define(
        cls,
        frames: Iterable[int],
        max_invalid_agents: int = 0,
        max_invalid_fraction: float = 0.0,
    ) -> RequireAgentCoverageAtFrames:
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


@dataclasses.dataclass(slots=True)
class MinimumAgentSamples(AgentValidationRule):
    """Require a minimum number of samples per retained agent."""

    minimum: int = Field(ge=1)
    max_invalid_agents: int = Field(default=0, ge=0)
    max_invalid_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    type: Literal["min_agent_samples"] = Field(
        "min_agent_samples",
        repr=False,
        init=False,
    )

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        return ctx.over_agent_window(pl.len()) >= self.minimum


AgentValidationSpec = Annotated[
    RequireCompleteAgentCoverage | RequireAgentCoverageAtFrames | MinimumAgentSamples,
    Field(discriminator="type"),
]
