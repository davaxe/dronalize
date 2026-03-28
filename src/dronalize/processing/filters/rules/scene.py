from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Annotated, Literal

import polars as pl
from pydantic import BeforeValidator, Field, dataclasses
from typing_extensions import override

from dronalize.core.categories import AgentCategory
from dronalize.processing.filters.rules.base import SceneFilterRule

if TYPE_CHECKING:
    from dronalize.processing.filters.context import FilterContext

AgentValue = int | str | AgentCategory
AgentInput = AgentValue | Iterable[AgentValue]
FrameInput = int | Iterable[int]


def _coerce_frame_set(value: FrameInput) -> FrameSet:
    """Convert one or many frame indices into a normalized integer set."""
    values = [value] if isinstance(value, int) else value
    return frozenset(int(item) for item in values)


FrameSet = Annotated[frozenset[int], Field(min_length=1), BeforeValidator(_coerce_frame_set)]


@dataclasses.dataclass(slots=True, frozen=True)
class MinimumAgents(SceneFilterRule):
    """Require a minimum number of retained agents in the scene."""

    minimum: int = Field(default=1, ge=0)
    type: Literal["minimum_agents"] = Field("minimum_agents", repr=False, init=False)

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        return ctx.retained_agent_count() >= self.minimum


@dataclasses.dataclass(slots=True, frozen=True)
class RequireSceneFrames(SceneFilterRule):
    """Require specific relative frames to exist in the scene window."""

    frames: FrameSet
    type: Literal["require_scene_frames"] = Field("require_scene_frames", repr=False, init=False)

    @classmethod
    def define(cls, frames: FrameInput) -> RequireSceneFrames:
        """Alternate constructor that accepts one or many frame indices."""
        return cls(frames=_coerce_frame_set(frames))

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        relative_frame = ctx.relative_frame()
        required = relative_frame.filter(relative_frame.is_in(self.frames)).n_unique()
        return ctx.over_scene_window(required) == len(self.frames)


@dataclasses.dataclass(slots=True, frozen=True)
class RequireContiguousSceneFrames(SceneFilterRule):
    """Require the cleaned scene window to have no frame gaps."""

    type: Literal["require_contiguous_scene_frames"] = Field(
        "require_contiguous_scene_frames", repr=False, init=False
    )

    @override
    def expr(self, ctx: FilterContext) -> pl.Expr:
        unique_frame_count = ctx.over_scene_window(pl.col(ctx.frame_column).n_unique())
        frame_span = ctx.over_scene_window(
            pl.col(ctx.frame_column).max() - pl.col(ctx.frame_column).min() + 1,
        )
        return unique_frame_count == frame_span
