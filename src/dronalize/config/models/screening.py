"""Models for declaratively specifying screening rules and cleanup actions."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import override

from dronalize.config.base import ConfigPatch, ResolvedConfig
from dronalize.core.categories import AgentCategoryLike

Categories = tuple[AgentCategoryLike, ...]
"""Tuple of category selectors accepted by screening rules."""


class AgentSelectorConfig(ResolvedConfig):
    """Category selection mode applied before evaluating a rule.

    `include` keeps only listed categories for the rule evaluation, while
    `exclude` applies the rule to all categories except the listed ones.
    """

    mode: Literal["include", "exclude"] = Field("include")
    """Whether listed categories are included in or excluded from the rule evaluation."""
    categories: Categories
    """Categories included in or excluded from the rule evaluation."""


class Tolerance(ResolvedConfig):
    """Optional tolerance thresholds for relaxed rule checks."""

    absolute: float | None = Field(default=None, gt=0.0)
    """Absolute tolerance applied when evaluating numeric thresholds."""
    relative: float | None = Field(default=None, gt=0.0, le=1.0)
    """Relative tolerance applied when evaluating numeric thresholds."""

    @model_validator(mode="after")
    def _require_tolerance(self) -> Tolerance:
        if self.absolute is None and self.relative is None:
            msg = "at least one of absolute or relative must be set."
            raise ValueError(msg)
        return self


class PassingRequirement(ResolvedConfig):
    """Minimum selected-agent pass thresholds for agent-rule scene acceptance."""

    absolute: int | None = Field(default=None, ge=1)
    """Absolute number of selected agents that must pass the rule."""
    relative: float | None = Field(default=None, gt=0.0, le=1.0)
    """Relative fraction of selected agents that must pass the rule."""

    @model_validator(mode="after")
    def _require_threshold(self) -> PassingRequirement:
        if self.absolute is None and self.relative is None:
            msg = "at least one of absolute or relative must be set."
            raise ValueError(msg)
        return self


class CountRange(ResolvedConfig):
    """Generic integer range with optional minimum and maximum."""

    minimum: int | None = None
    """Inclusive lower bound for the range, if one is required."""
    maximum: int | None = None
    """Inclusive upper bound for the range, if one is required."""

    @model_validator(mode="after")
    def _val_range(self) -> CountRange:
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            msg = "maximum must be greater than or equal to minimum."
            raise ValueError(msg)
        return self


class _AgentRuleSpecBase(ResolvedConfig):
    """Shared options for per-agent screening rules."""

    selector: AgentSelectorConfig | None = Field(default=None)
    """Optional category selector limiting which agents the rule evaluates."""
    tolerance: Tolerance | None = Field(default=None)
    """Optional tolerance thresholds used to relax rule comparisons."""
    require: PassingRequirement | None = Field(default=None)
    """Optional minimum selected-agent pass thresholds required to keep the scene."""


class MinDistanceSpec(_AgentRuleSpecBase):
    """Require a minimum distance between first and last observations."""

    rule: Literal["min_distance"] = Field("min_distance", repr=False, init=False)
    minimum: float = Field(ge=0.0)
    """Minimum distance between first and last observations for each selected agent."""


class RequireFramesSpec(_AgentRuleSpecBase):
    """Require listed frame IDs to exist for each selected agent."""

    rule: Literal["frames"] = Field("frames", repr=False, init=False)
    frames: tuple[int, ...]
    """Frame indices that must be present for each selected agent."""


class RequireWindowSpec(_AgentRuleSpecBase):
    """Require agent coverage within an inclusive frame window."""

    rule: Literal["window"] = Field("window", repr=False, init=False)
    start_frame: int = Field(ge=0)
    """Inclusive start frame of the required coverage window."""
    end_frame: int = Field(ge=0)
    """Inclusive end frame of the required coverage window."""
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    """Minimum fraction of frames within the window that must be present."""


class MinSamplesSpec(_AgentRuleSpecBase):
    """Require a minimum number of samples for each selected agent."""

    rule: Literal["min_samples"] = Field("min_samples", repr=False, init=False)
    minimum: int = Field(ge=1)
    """Minimum number of samples required for each selected agent."""


class MaxMissingFramesSpec(_AgentRuleSpecBase):
    """Limit how many frames may be missing per selected agent."""

    rule: Literal["max_missing_frames"] = Field("max_missing_frames", repr=False, init=False)
    maximum: int = Field(ge=0)
    """Maximum number of missing frames allowed for each selected agent."""


class MaxGapSpec(_AgentRuleSpecBase):
    """Limit the longest consecutive missing gap per selected agent."""

    rule: Literal["max_gap"] = Field("max_gap", repr=False, init=False)
    maximum: int = Field(ge=0)
    """Maximum consecutive missing-frame gap allowed for each selected agent."""


class MinConsecutiveFramesSpec(_AgentRuleSpecBase):
    """Require a minimum uninterrupted run of valid frames."""

    rule: Literal["min_consecutive_frames"] = Field(
        "min_consecutive_frames", repr=False, init=False
    )
    minimum: int = Field(ge=1)
    """Minimum uninterrupted run of valid frames required per selected agent."""


class StartsByFrameSpec(_AgentRuleSpecBase):
    """Require each selected agent to start at or before a frame index."""

    rule: Literal["starts_by_frame"] = Field("starts_by_frame", repr=False, init=False)
    frame: int = Field(ge=0)
    """Latest frame index by which each selected agent must have started."""


class EndsAfterFrameSpec(_AgentRuleSpecBase):
    """Require each selected agent to continue past a frame index."""

    rule: Literal["ends_after_frame"] = Field("ends_after_frame", repr=False, init=False)
    frame: int = Field(ge=0)
    """Frame index after which each selected agent must still be present."""


class MinSpanSpec(_AgentRuleSpecBase):
    """Require a minimum frame-span between first and last observations."""

    rule: Literal["min_span"] = Field("min_span", repr=False, init=False)
    minimum: int = Field(ge=1)
    """Minimum frame span between first and last observations for each selected agent."""


AgentCheckSpec = Annotated[
    RequireFramesSpec
    | RequireWindowSpec
    | MinSamplesSpec
    | MaxMissingFramesSpec
    | MaxGapSpec
    | MinConsecutiveFramesSpec
    | StartsByFrameSpec
    | EndsAfterFrameSpec
    | MinSpanSpec
    | MinDistanceSpec,
    Field(discriminator="rule"),
]
"""Discriminated union of all declarative per-agent screening rule models.

Each variant describes one rule family that can be placed in the `agent`
section of a
[`ScreeningConfig`][dronalize.config.models.screening.ScreeningConfig].
"""


class AgentRangeSpec(ResolvedConfig):
    """Constrain the count of selected agents present in a scene."""

    rule: Literal["agent_range"] = Field("agent_range", repr=False, init=False)
    selector: AgentSelectorConfig | None = None
    """Optional category selector limiting which agents are counted."""
    minimum: int | None = Field(default=None, ge=0)
    """Inclusive lower bound on the number of matching agents in a scene."""
    maximum: int | None = Field(default=None, ge=0)
    """Inclusive upper bound on the number of matching agents in a scene."""


class CategoryRangeSpec(ResolvedConfig):
    """Constrain per-category agent counts using named integer ranges."""

    rule: Literal["category_range"] = Field("category_range", repr=False, init=False)
    ranges: dict[str, CountRange]
    """Required count ranges keyed by category name."""


class RequireSceneFramesSpec(ResolvedConfig):
    """Require specific frame IDs to exist at the scene level."""

    rule: Literal["scene_frames"] = Field("scene_frames", repr=False, init=False)
    frames: tuple[int, ...]
    """Frame indices that must exist at scene scope."""


class RequireSceneWindowSpec(ResolvedConfig):
    """Require scene occupancy over an inclusive frame window."""

    rule: Literal["scene_window"] = Field("scene_window", repr=False, init=False)
    start_frame: int = Field(ge=0)
    """Inclusive start frame of the required scene coverage window."""
    end_frame: int = Field(ge=0)
    """Inclusive end frame of the required scene coverage window."""
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    """Minimum fraction of scene frames within the window that must exist."""


class MaxMissingSceneFramesSpec(ResolvedConfig):
    """Limit missing frames at scene scope for a selected category set."""

    rule: Literal["max_missing_frames"] = Field("max_missing_frames", repr=False, init=False)
    selector: AgentSelectorConfig
    """Category selector defining which agents contribute to the scene-level check."""
    maximum: int = Field(ge=0)
    """Maximum number of missing scene frames allowed for the selected agents."""


SceneCheckSpec = Annotated[
    AgentRangeSpec
    | CategoryRangeSpec
    | RequireSceneFramesSpec
    | RequireSceneWindowSpec
    | MaxMissingSceneFramesSpec,
    Field(discriminator="rule"),
]
"""Discriminated union of all declarative scene-level screening rule models.

These models power the `scene` section of
[`ScreeningConfig`][dronalize.config.models.screening.ScreeningConfig].
"""


class PruneByRuleSpec(ResolvedConfig):
    """Cleanup action that removes agents matching a nested agent rule."""

    rule: Literal["prune_by"] = Field("prune_by", repr=False, init=False)
    agent_rule: AgentCheckSpec
    """Nested agent rule used to remove matching agents from the scene."""

    @model_validator(mode="after")
    def _reject_aggregate_requirements(self) -> PruneByRuleSpec:
        if self.agent_rule.require is not None:
            msg = "`require` is only valid for agent screening rules, not cleanup pruning."
            raise ValueError(msg)
        return self


class ExcludeCategoriesSpec(ResolvedConfig):
    """Cleanup action that drops listed categories from a scene."""

    rule: Literal["exclude"] = Field("exclude", repr=False, init=False)
    categories: Categories
    """Categories removed from the scene during cleanup."""


class IncludeCategoriesSpec(ResolvedConfig):
    """Cleanup action that keeps only listed categories in a scene."""

    rule: Literal["include"] = Field("include", repr=False, init=False)
    categories: Categories
    """Categories retained in the scene during cleanup."""


CleanupSpec = Annotated[
    ExcludeCategoriesSpec | IncludeCategoriesSpec | PruneByRuleSpec, Field(discriminator="rule")
]
"""Discriminated union of all declarative cleanup actions applied before screening.

Cleanup specs can drop categories or prune agents based on nested screening
rules before the main scene and agent checks run.
"""


class ScreeningConfig(ResolvedConfig):
    """Declarative screening configuration composed from named rule maps."""

    cleanup: dict[str, CleanupSpec] = Field(default_factory=dict)
    """Named cleanup actions applied before screening checks are evaluated."""
    scene: dict[str, SceneCheckSpec] = Field(default_factory=dict)
    """Named scene-level screening rules."""
    agent: dict[str, AgentCheckSpec] = Field(default_factory=dict)
    """Named agent-level screening rules."""


class PartialScreeningConfig(ConfigPatch[ScreeningConfig]):
    """Patch model for replacing or extending named screening rule sets."""

    mode: Literal["replace", "extend"] | None = None
    """How this partial screening config should combine with an existing target."""
    remove: tuple[str, ...] | None = None
    """Named cleanup, scene, or agent rules to remove after merging."""
    cleanup: dict[str, CleanupSpec] | None = None
    """Cleanup rule overrides keyed by rule name."""
    scene: dict[str, SceneCheckSpec] | None = None
    """Scene-level rule overrides keyed by rule name."""
    agent: dict[str, AgentCheckSpec] | None = None
    """Agent-level rule overrides keyed by rule name."""
    full_config_type: type[ScreeningConfig] = Field(ScreeningConfig, repr=False, init=False)

    @override
    def merge_into(
        self, target: ScreeningConfig | None, *, exclude_none: bool = True
    ) -> ScreeningConfig:
        mode = self.mode or "extend"  # defaults to extend
        cleanup = target.cleanup if target is not None else {}
        scene = target.scene if target is not None else {}
        agent = target.agent if target is not None else {}
        if mode == "replace":
            cleanup = self.cleanup if self.cleanup is not None else {}
            scene = self.scene if self.scene is not None else {}
            agent = self.agent if self.agent is not None else {}
        elif mode == "extend":
            cleanup = {**cleanup, **(self.cleanup or {})}
            scene = {**scene, **(self.scene or {})}
            agent = {**agent, **(self.agent or {})}
        if self.remove:
            cleanup = {k: v for k, v in cleanup.items() if k not in self.remove}
            scene = {k: v for k, v in scene.items() if k not in self.remove}
            agent = {k: v for k, v in agent.items() if k not in self.remove}
        return ScreeningConfig(cleanup=cleanup, scene=scene, agent=agent)
