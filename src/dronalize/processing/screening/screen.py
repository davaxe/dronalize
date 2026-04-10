"""Screen containers, declarative configs, and rule-compilation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, TypeVar

from pydantic import BaseModel, BeforeValidator

from dronalize.processing.screening.agent import AgentCheckRule
from dronalize.processing.screening.base import AgentCheckRuleBase, Rule
from dronalize.processing.screening.cleanup import CleanupRule, PruneByRule
from dronalize.processing.screening.scene import SceneCheckRule

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.config.sections import ScreeningConfig


RuleT = TypeVar("RuleT", bound=Rule)
RuleSpecT = TypeVar("RuleSpecT", bound=BaseModel)
AgentCheckSpecs = Annotated[tuple[AgentCheckRule, ...], BeforeValidator(tuple)]
CleanupSpecs = Annotated[tuple[CleanupRule, ...], BeforeValidator(tuple)]
SceneCheckSpecs = Annotated[tuple[SceneCheckRule, ...], BeforeValidator(tuple)]


@dataclass(slots=True, frozen=True)
class Screen:
    """Collection of cleanup and check rules used during screening."""

    cleanup_rules: tuple[CleanupRule, ...] = ()
    scene_rules: tuple[SceneCheckRule, ...] = ()
    agent_rules: tuple[AgentCheckRule, ...] = ()

    def __post_init__(self) -> None:
        """Validate that all rules have unique names."""
        seen: set[str] = set()
        for rule in (*self.cleanup_rules, *self.scene_rules, *self.agent_rules):
            name = rule.name()
            if name in seen:
                msg = f"Duplicate rule name: {name}"
                raise ValueError(msg)
            seen.add(name)

    @classmethod
    def define(
        cls,
        cleanup_rules: Iterable[CleanupRule | AgentCheckRule] = (),
        scene_rules: Iterable[SceneCheckRule] = (),
        agent_rules: Iterable[AgentCheckRule] = (),
    ) -> Screen:
        """Return a new Screen instance with the given rules, validating uniqueness."""
        rules: list[CleanupRule] = []
        for rule in cleanup_rules:
            if isinstance(rule, AgentCheckRuleBase):
                rules.append(PruneByRule(agent_rule=rule))
            else:
                rules.append(rule)
        return cls(
            cleanup_rules=tuple(rules),
            scene_rules=tuple(scene_rules),
            agent_rules=tuple(agent_rules),
        )

    @classmethod
    def from_config(cls, config: ScreeningConfig) -> Screen:
        """Return a Screen instance compiled from a ScreeningConfig."""
        return cls.define(
            cleanup_rules=_CleanupRuleCompiler.compile(config.cleanup),
            scene_rules=_SceneCheckRuleCompiler.compile(config.scene),
            agent_rules=_AgentCheckRuleCompiler.compile(config.agent),
        )


class _AgentCheckRuleCompiler(BaseModel):
    value: AgentCheckRule

    @classmethod
    def compile(cls, entries: dict[str, RuleSpecT]) -> tuple[AgentCheckRule, ...]:
        compiled: list[AgentCheckRule] = []
        for name, spec in entries.items():
            rule = cls.model_validate(spec.model_dump())
            rule = rule.model_copy(update={"rule_id": name, "enabled": True})
            compiled.append(rule.value)
        return tuple(compiled)


class _SceneCheckRuleCompiler(BaseModel):
    value: SceneCheckRule

    @classmethod
    def compile(cls, entries: dict[str, RuleSpecT]) -> tuple[SceneCheckRule, ...]:
        compiled: list[SceneCheckRule] = []
        for name, spec in entries.items():
            rule = cls.model_validate(spec.model_dump())
            rule = rule.model_copy(update={"rule_id": name, "enabled": True})
            compiled.append(rule.value)
        return tuple(compiled)


class _CleanupRuleCompiler(BaseModel):
    value: CleanupRule

    @classmethod
    def compile(cls, entries: dict[str, RuleSpecT]) -> tuple[CleanupRule, ...]:
        compiled: list[CleanupRule] = []
        for name, spec in entries.items():
            rule = cls.model_validate(spec.model_dump())
            rule = rule.model_copy(update={"rule_id": name, "enabled": True})
            compiled.append(rule.value)
        return tuple(compiled)
