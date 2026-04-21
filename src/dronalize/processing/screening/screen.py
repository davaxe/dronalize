"""Screen containers, declarative configs, and rule-compilation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Generic, TypeVar

import rich
from pydantic import BaseModel, BeforeValidator, TypeAdapter
from typing_extensions import override

from dronalize.processing.screening.agent import AgentCheckRule
from dronalize.processing.screening.base import AgentCheckRuleBase, Rule
from dronalize.processing.screening.cleanup import CleanupRule, PruneByRule
from dronalize.processing.screening.scene import SceneCheckRule

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.config.models import ScreeningConfig


_RuleT = TypeVar("_RuleT", bound=Rule)
_RuleSpecT = TypeVar("_RuleSpecT", bound=BaseModel)
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


class _RuleCompiler(ABC, Generic[_RuleT]):
    @classmethod
    @abstractmethod
    def adapter(cls) -> TypeAdapter[_RuleT]:
        """Return the TypeAdapter used to validate rule specs."""
        ...

    @classmethod
    def compile(cls, entries: dict[str, _RuleSpecT]) -> tuple[_RuleT, ...]:
        compiled: list[_RuleT] = []
        for name, spec in entries.items():
            payload = spec.model_dump(exclude_none=True)
            rule = cls.adapter().validate_python(payload)
            rule = rule.model_copy(update={"rule_id": name, "enabled": True})
            compiled.append(rule)
        return tuple(compiled)


class _CleanupRuleCompiler(_RuleCompiler[CleanupRule]):
    @classmethod
    @override
    def adapter(cls) -> TypeAdapter[CleanupRule]:
        """Return the TypeAdapter used to validate cleanup rule specs."""
        return TypeAdapter(CleanupRule)


class _SceneCheckRuleCompiler(_RuleCompiler[SceneCheckRule]):
    @classmethod
    @override
    def adapter(cls) -> TypeAdapter[SceneCheckRule]:
        """Return the TypeAdapter used to validate scene check rule specs."""
        return TypeAdapter(SceneCheckRule)


class _AgentCheckRuleCompiler(_RuleCompiler[AgentCheckRule]):
    @classmethod
    @override
    def adapter(cls) -> TypeAdapter[AgentCheckRule]:
        """Return the TypeAdapter used to validate agent check rule specs."""
        return TypeAdapter(AgentCheckRule)
