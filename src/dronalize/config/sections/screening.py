from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator
from typing_extensions import override

from dronalize.config.base import FullConfig, PartialConfig
from dronalize.core.categories import AgentCategoryLike

Categories = tuple[AgentCategoryLike, ...]


class AgentSelector(FullConfig):
    mode: Literal["include", "exclude"] = Field("include")
    categories: Categories


class Tolerance(FullConfig):
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

    def min_max(self) -> tuple[int | None, int | None]:
        """Return minimum and maximum as a tuple."""
        return self.minimum, self.maximum


class _AgentRuleSpecBase(FullConfig):
    selector: AgentSelector | None = Field(default=None)
    tolerance: Tolerance | None = Field(default=None)


class RequireFramesSpec(_AgentRuleSpecBase):
    rule: Literal["frames"] = Field("frames", repr=False, init=False)
    frames: tuple[int, ...]


class RequireWindowSpec(_AgentRuleSpecBase):
    rule: Literal["window"] = Field("window", repr=False, init=False)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)


class MinSamplesSpec(_AgentRuleSpecBase):
    rule: Literal["min_samples"] = Field("min_samples", repr=False, init=False)
    minimum: int = Field(ge=1)


class MaxMissingFramesSpec(_AgentRuleSpecBase):
    rule: Literal["max_missing_frames"] = Field("max_missing_frames", repr=False, init=False)
    maximum: int = Field(ge=0)


class MaxGapSpec(_AgentRuleSpecBase):
    rule: Literal["max_gap"] = Field("max_gap", repr=False, init=False)
    maximum: int = Field(ge=0)


class MinConsecutiveFramesSpec(_AgentRuleSpecBase):
    rule: Literal["min_consecutive_frames"] = Field(
        "min_consecutive_frames", repr=False, init=False
    )
    minimum: int = Field(ge=1)


class StartsByFrameSpec(_AgentRuleSpecBase):
    rule: Literal["starts_by_frame"] = Field("starts_by_frame", repr=False, init=False)
    frame: int = Field(ge=0)


class EndsAfterFrameSpec(_AgentRuleSpecBase):
    rule: Literal["ends_after_frame"] = Field("ends_after_frame", repr=False, init=False)
    frame: int = Field(ge=0)


class MinSpanSpec(_AgentRuleSpecBase):
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
    rule: Literal["agent_range"] = Field("agent_range", repr=False, init=False)
    selector: AgentSelector | None = None
    minimum: int | None = Field(default=None, ge=0)
    maximum: int | None = Field(default=None, ge=0)


class CategoryRangeSpec(FullConfig):
    rule: Literal["category_range"] = Field("category_range", repr=False, init=False)
    ranges: dict[str, Range]


class RequireSceneFramesSpec(FullConfig):
    rule: Literal["scene_frames"] = Field("scene_frames", repr=False, init=False)
    frames: tuple[int, ...]


class RequireSceneWindowSpec(FullConfig):
    rule: Literal["scene_window"] = Field("scene_window", repr=False, init=False)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)


class MaxMissingSceneFramesSpec(FullConfig):
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
    rule: Literal["prune_by"] = Field("prune_by", repr=False, init=False)
    agent_rule: AgentCheckSpec


class ExcludeCategoriesSpec(FullConfig):
    rule: Literal["exclude"] = Field("exclude", repr=False, init=False)
    categories: Categories


class IncludeCategoriesSpec(FullConfig):
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
