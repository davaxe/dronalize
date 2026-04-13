from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import override

from dronalize.config.base import FullConfig, PartialConfig
from dronalize.core.categories import AgentCategoryLike

Categories = tuple[AgentCategoryLike, ...]
"""Tuple of category selectors accepted by screening rules."""


class AgentSelector(FullConfig):
    """Category selection mode applied before evaluating a rule.

    `include` keeps only listed categories for the rule evaluation, while
    `exclude` applies the rule to all categories except the listed ones.
    """

    mode: Literal["include", "exclude"] = Field("include")
    categories: Categories


class Tolerance(FullConfig):
    """Optional tolerance thresholds for relaxed rule checks."""

    absolute: float | None = Field(default=None, gt=0.0)
    relative: float | None = Field(default=None, gt=0.0)


class Range(FullConfig):
    """Generic integer range with optional minimum and maximum."""

    minimum: int | None = None
    maximum: int | None = None

    @model_validator(mode="after")
    def _val_range(self) -> Range:
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            msg = "maximum must be greater than or equal to minimum."
            raise ValueError(msg)
        return self


class _AgentRuleSpecBase(FullConfig):
    """Shared options for per-agent screening rules."""

    selector: AgentSelector | None = Field(default=None)
    tolerance: Tolerance | None = Field(default=None)


class RequireFramesSpec(_AgentRuleSpecBase):
    """Require listed frame IDs to exist for each selected agent."""

    rule: Literal["frames"] = Field("frames", repr=False, init=False)
    frames: tuple[int, ...]


class RequireWindowSpec(_AgentRuleSpecBase):
    """Require agent coverage within an inclusive frame window."""

    rule: Literal["window"] = Field("window", repr=False, init=False)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)


class MinSamplesSpec(_AgentRuleSpecBase):
    """Require a minimum number of samples for each selected agent."""

    rule: Literal["min_samples"] = Field("min_samples", repr=False, init=False)
    minimum: int = Field(ge=1)


class MaxMissingFramesSpec(_AgentRuleSpecBase):
    """Limit how many frames may be missing per selected agent."""

    rule: Literal["max_missing_frames"] = Field("max_missing_frames", repr=False, init=False)
    maximum: int = Field(ge=0)


class MaxGapSpec(_AgentRuleSpecBase):
    """Limit the longest consecutive missing gap per selected agent."""

    rule: Literal["max_gap"] = Field("max_gap", repr=False, init=False)
    maximum: int = Field(ge=0)


class MinConsecutiveFramesSpec(_AgentRuleSpecBase):
    """Require a minimum uninterrupted run of observed frames."""

    rule: Literal["min_consecutive_frames"] = Field(
        "min_consecutive_frames", repr=False, init=False
    )
    minimum: int = Field(ge=1)


class StartsByFrameSpec(_AgentRuleSpecBase):
    """Require each selected agent to start at or before a frame index."""

    rule: Literal["starts_by_frame"] = Field("starts_by_frame", repr=False, init=False)
    frame: int = Field(ge=0)


class EndsAfterFrameSpec(_AgentRuleSpecBase):
    """Require each selected agent to continue past a frame index."""

    rule: Literal["ends_after_frame"] = Field("ends_after_frame", repr=False, init=False)
    frame: int = Field(ge=0)


class MinSpanSpec(_AgentRuleSpecBase):
    """Require a minimum frame-span between first and last observations."""

    rule: Literal["min_span"] = Field("min_span", repr=False, init=False)
    minimum: int = Field(ge=1)


AgentCheckSpec = Annotated[
    RequireFramesSpec
    | RequireWindowSpec
    | MinSamplesSpec
    | MaxMissingFramesSpec
    | MaxGapSpec
    | MinConsecutiveFramesSpec
    | StartsByFrameSpec
    | EndsAfterFrameSpec
    | MinSpanSpec,
    Field(discriminator="rule"),
]


class AgentRangeSpec(FullConfig):
    """Constrain the count of selected agents present in a scene."""

    rule: Literal["agent_range"] = Field("agent_range", repr=False, init=False)
    selector: AgentSelector | None = None
    minimum: int | None = Field(default=None, ge=0)
    maximum: int | None = Field(default=None, ge=0)


class CategoryRangeSpec(FullConfig):
    """Constrain per-category agent counts using named integer ranges."""

    rule: Literal["category_range"] = Field("category_range", repr=False, init=False)
    ranges: dict[str, Range]


class RequireSceneFramesSpec(FullConfig):
    """Require specific frame IDs to exist at the scene level."""

    rule: Literal["scene_frames"] = Field("scene_frames", repr=False, init=False)
    frames: tuple[int, ...]


class RequireSceneWindowSpec(FullConfig):
    """Require scene occupancy over an inclusive frame window."""

    rule: Literal["scene_window"] = Field("scene_window", repr=False, init=False)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)


class MaxMissingSceneFramesSpec(FullConfig):
    """Limit missing frames at scene scope for a selected category set."""

    rule: Literal["max_missing_frames"] = Field("max_missing_frames", repr=False, init=False)
    selector: AgentSelector
    maximum: int = Field(ge=0)


SceneCheckSpec = Annotated[
    AgentRangeSpec
    | CategoryRangeSpec
    | RequireSceneFramesSpec
    | RequireSceneWindowSpec
    | MaxMissingSceneFramesSpec,
    Field(discriminator="rule"),
]


class PruneByRuleSpec(FullConfig):
    """Cleanup action that removes agents matching a nested agent rule."""

    rule: Literal["prune_by"] = Field("prune_by", repr=False, init=False)
    agent_rule: AgentCheckSpec


class ExcludeCategoriesSpec(FullConfig):
    """Cleanup action that drops listed categories from a scene."""

    rule: Literal["exclude"] = Field("exclude", repr=False, init=False)
    categories: Categories


class IncludeCategoriesSpec(FullConfig):
    """Cleanup action that keeps only listed categories in a scene."""

    rule: Literal["include"] = Field("include", repr=False, init=False)
    categories: Categories


CleanupSpec = Annotated[
    ExcludeCategoriesSpec | IncludeCategoriesSpec | PruneByRuleSpec,
    Field(discriminator="rule"),
]


class ScreeningConfig(FullConfig):
    """Declarative screening configuration composed from named rule maps."""

    cleanup: dict[str, CleanupSpec] = Field(default_factory=dict)
    scene: dict[str, SceneCheckSpec] = Field(default_factory=dict)
    agent: dict[str, AgentCheckSpec] = Field(default_factory=dict)


class PartialScreeningConfig(PartialConfig[ScreeningConfig]):
    """Patch model for replacing or extending named screening rule sets."""

    mode: Literal["replace", "extend"] | None = None
    remove: tuple[str, ...] | None = None
    cleanup: dict[str, CleanupSpec] | None = None
    scene: dict[str, SceneCheckSpec] | None = None
    agent: dict[str, AgentCheckSpec] | None = None
    full_config_type: type[ScreeningConfig] = ScreeningConfig

    @override
    def apply_to(
        self, target: ScreeningConfig | None, *, exclude_none: bool = True
    ) -> ScreeningConfig:
        mode = self.mode or "replace"
        cleanup = target.cleanup if target is not None else {}
        scene = target.scene if target is not None else {}
        agent = target.agent if target is not None else {}
        if mode == "replace":
            cleanup = self.cleanup if self.cleanup is not None else {}
            scene = self.scene if self.scene is not None else {}
            agent = self.agent if self.agent is not None else {}
        elif mode == "extend" and target is not None:
            cleanup = {**target.cleanup, **(self.cleanup or {})}
            scene = {**target.scene, **(self.scene or {})}
            agent = {**target.agent, **(self.agent or {})}
        if self.remove:
            cleanup = {k: v for k, v in cleanup.items() if k not in self.remove}
            scene = {k: v for k, v in scene.items() if k not in self.remove}
            agent = {k: v for k, v in agent.items() if k not in self.remove}
        return ScreeningConfig(cleanup=cleanup, scene=scene, agent=agent)
