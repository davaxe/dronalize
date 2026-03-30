from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, dataclasses, model_validator

from dronalize.processing.filters.agent import AgentCheckSpec
from dronalize.processing.filters.base import AgentCheckRule, CleanupRule, Rule, SceneCheckRule
from dronalize.processing.filters.cleanup import CleanupSpec, PruneByRule
from dronalize.processing.filters.scene import SceneCheckSpec

if TYPE_CHECKING:
    from collections.abc import Iterable


RuleT = TypeVar("RuleT", bound=Rule)
AgentCheckSpecs = Annotated[tuple[AgentCheckSpec, ...], BeforeValidator(tuple)]
CleanupSpecs = Annotated[tuple[CleanupSpec, ...], BeforeValidator(tuple)]
SceneCheckSpecs = Annotated[tuple[SceneCheckSpec, ...], BeforeValidator(tuple)]


@dataclasses.dataclass(slots=True, frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Filter:
    """Collection of cleanup and check rules."""

    cleanup_rules: tuple[CleanupRule, ...] = ()
    scene_rules: tuple[SceneCheckRule, ...] = ()
    agent_rules: tuple[AgentCheckRule, ...] = ()

    @model_validator(mode="after")
    def _validate_uniqueness(self) -> Filter:
        # Since the name is used to determine possible temporary columns in
        # dataframe transformations, it must be unique across all rules.
        seen: set[str] = set()
        for rule in (*self.cleanup_rules, *self.scene_rules, *self.agent_rules):
            name = rule.name()
            if name in seen:
                msg = f"Duplicate rule name: {name}"
                raise ValueError(msg)
            seen.add(name)
        return self

    @classmethod
    def define(
        cls,
        cleanup_rules: Iterable[CleanupRule] = (),
        scene_rules: Iterable[SceneCheckRule] = (),
        agent_rules: Iterable[AgentCheckRule] = (),
    ) -> Filter:
        """Return a new Filter instance with the given rules, validating uniqueness."""
        return cls(
            cleanup_rules=tuple(cleanup_rules),
            scene_rules=tuple(scene_rules),
            agent_rules=tuple(agent_rules),
        )

    @classmethod
    def define_cleanup(cls, *rules: CleanupRule | AgentCheckRule) -> Filter:
        """Return a new Filter instance with only cleanup rules.

        Also accepts `AgentCheckRule`. These will be wrapped in
        `PruneByRule` to be applied as cleanup rules that remove entire
        agents instead of individual rows.

        """
        rules_list: list[CleanupRule] = []
        for rule in rules:
            if isinstance(rule, AgentCheckRule):
                rules_list.append(PruneByRule(rule=rule, rule_id=rule.rule_id))
                continue
            rules_list.append(rule)
        return cls.define(cleanup_rules=rules_list)

    @classmethod
    def define_scene_rules(cls, *rules: SceneCheckRule) -> Filter:
        """Return a new Filter instance with only scene check rules."""
        return cls.define(scene_rules=rules)

    @classmethod
    def define_agent_rules(cls, *rules: AgentCheckRule) -> Filter:
        """Return a new Filter instance with only agent check rules."""
        return cls.define(agent_rules=rules)


class FilterSpec(BaseModel):
    """Config-facing filter specification that resolves into a runtime `Filter`."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["replace", "merge"] = "replace"
    cleanup: CleanupSpecs = Field(())
    scene: SceneCheckSpecs = Field(())
    agent: AgentCheckSpecs = Field(())

    def resolve(self, base: Filter | None = None) -> Filter:
        """Resolve the spec into the executable runtime filter object."""
        if self.mode == "replace":
            return Filter.define(
                cleanup_rules=self.cleanup, scene_rules=self.scene, agent_rules=self.agent
            )

        base_filter = Filter() if base is None else base
        return Filter.define(
            cleanup_rules=_merge_rules(base_filter.cleanup_rules, self.cleanup),
            scene_rules=_merge_rules(base_filter.scene_rules, self.scene),
            agent_rules=_merge_rules(base_filter.agent_rules, self.agent),
        )


def _merge_rules(base: tuple[RuleT, ...], updates: tuple[RuleT, ...]) -> tuple[RuleT, ...]:
    """Merge rules by effective rule key while preserving their visible order."""
    if not updates:
        return base

    updates_by_name = {rule.name(): rule for rule in updates}
    merged = [updates_by_name.pop(rule.name(), rule) for rule in base]
    merged.extend(updates_by_name.values())
    return tuple(merged)
