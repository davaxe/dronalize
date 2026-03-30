from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, TypeVar

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    dataclasses,
    model_validator,
)

from dronalize.processing.filters.rules.agent import AgentValidationSpec
from dronalize.processing.filters.rules.base import (
    AgentValidationRule,
    CleanupRule,
    Rule,
    SceneValidationRule,
)
from dronalize.processing.filters.rules.cleanup import CleanupSpec
from dronalize.processing.filters.rules.scene import SceneValidationSpec

if TYPE_CHECKING:
    from collections.abc import Iterable


RuleT = TypeVar("RuleT", bound=Rule)
AgentValidationSpecs = Annotated[tuple[AgentValidationSpec, ...], BeforeValidator(tuple)]
CleanupSpecs = Annotated[tuple[CleanupSpec, ...], BeforeValidator(tuple)]
SceneValidationSpecs = Annotated[tuple[SceneValidationSpec, ...], BeforeValidator(tuple)]


@dataclasses.dataclass(slots=True, frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Filter:
    """Collection of cleanup and validation rules."""

    cleanup_rules: tuple[CleanupRule, ...] = ()
    scene_validation_rules: tuple[SceneValidationRule, ...] = ()
    agent_validation_rules: tuple[AgentValidationRule, ...] = ()

    @model_validator(mode="after")
    def _validate_uniqueness(self) -> Filter:
        def _ensure_unique(rules: tuple[Rule, ...]) -> None:
            seen: set[str] = set()
            for rule in rules:
                name = rule.name()
                if name in seen:
                    msg = f"Duplicate rule name: {name}"
                    raise ValueError(msg)
                seen.add(name)

        _ensure_unique(self.cleanup_rules)
        _ensure_unique(self.scene_validation_rules)
        _ensure_unique(self.agent_validation_rules)
        return self

    @classmethod
    def define(
        cls,
        cleanup_rules: Iterable[CleanupRule] = (),
        scene_validation_rules: Iterable[SceneValidationRule] = (),
        agent_validation_rules: Iterable[AgentValidationRule] = (),
    ) -> Filter:
        """Return a new Filter instance with the given rules, validating uniqueness."""
        return cls(
            cleanup_rules=tuple(cleanup_rules),
            scene_validation_rules=tuple(scene_validation_rules),
            agent_validation_rules=tuple(agent_validation_rules),
        )


class FilterSpec(BaseModel):
    """Config-facing filter specification that resolves into a runtime `Filter`."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["replace", "merge"] = "replace"
    cleanup: CleanupSpecs = Field((), validation_alias=AliasChoices("cleanup", "cleanup_rules"))
    validate_scene: SceneValidationSpecs = Field(
        (), validation_alias=AliasChoices("validate_scene", "scene_validation_rules")
    )
    validate_agent: AgentValidationSpecs = Field(
        (), validation_alias=AliasChoices("validate_agent", "agent_validation_rules")
    )

    def resolve(self, base: Filter | None = None) -> Filter:
        """Resolve the spec into the executable runtime filter object."""
        if self.mode == "replace":
            return Filter.define(
                cleanup_rules=self.cleanup,
                scene_validation_rules=self.validate_scene,
                agent_validation_rules=self.validate_agent,
            )

        base_filter = Filter() if base is None else base
        return Filter.define(
            cleanup_rules=_merge_rules(base_filter.cleanup_rules, self.cleanup),
            scene_validation_rules=_merge_rules(
                base_filter.scene_validation_rules,
                self.validate_scene,
            ),
            agent_validation_rules=_merge_rules(
                base_filter.agent_validation_rules,
                self.validate_agent,
            ),
        )


def _merge_rules(base: tuple[RuleT, ...], updates: tuple[RuleT, ...]) -> tuple[RuleT, ...]:
    """Merge rules by stable rule name while preserving their visible order."""
    if not updates:
        return base

    updates_by_name = {rule.name(): rule for rule in updates}
    merged = [updates_by_name.pop(rule.name(), rule) for rule in base]
    merged.extend(updates_by_name.values())
    return tuple(merged)
