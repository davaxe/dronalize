from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, TypeVar, cast

from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    dataclasses,
)

from dronalize.processing.filters.rules.agent import (
    MinimumAgentSamples,
    RequireAgentFrames,
    RequireFullAgentWindow,
)
from dronalize.processing.filters.rules.base import (
    CleanupRule,
    FilterRule,
    Rule,
    rule_name,
)
from dronalize.processing.filters.rules.scene import (
    DropAgentCategories,
    MinimumAgents,
    RequireContiguousSceneFrames,
    RequireSceneFrames,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def _ensure_unique(rules: tuple[Rule, ...]) -> tuple[Rule, ...]:
    """Validate that no two rules share the same name."""
    seen: set[str] = set()
    for rule in rules:
        name = rule_name(rule)
        if name in seen:
            msg = f"Duplicate rule name: {name}"
            raise ValueError(msg)
        seen.add(name)
    return rules


RuleT = TypeVar("RuleT", bound=Rule)

CleanupRuleValue = DropAgentCategories
FilterRuleValue = Annotated[
    MinimumAgents
    | RequireSceneFrames
    | RequireContiguousSceneFrames
    | RequireFullAgentWindow
    | RequireAgentFrames
    | MinimumAgentSamples,
    Field(discriminator="type"),
]

CleanupRules = Annotated[
    tuple[CleanupRuleValue, ...], AfterValidator(_ensure_unique), BeforeValidator(tuple)
]
FilterRules = Annotated[
    tuple[FilterRuleValue, ...], AfterValidator(_ensure_unique), BeforeValidator(tuple)
]


@dataclasses.dataclass(slots=True, frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Filter:
    """Collection of cleanup and validation rules."""

    cleanup_rules: CleanupRules = ()
    filter_rules: FilterRules = ()

    @classmethod
    def define(
        cls,
        cleanup_rules: Iterable[CleanupRule] = (),
        filter_rules: Iterable[FilterRule] = (),
    ) -> Filter:
        """Return a new Filter instance with the given rules, validating uniqueness."""
        return cls(
            cleanup_rules=cast("CleanupRules", tuple(cleanup_rules)),
            filter_rules=cast("FilterRules", tuple(filter_rules)),
        )


class FilterSpec(BaseModel):
    """Config-facing filter specification that resolves into a runtime `Filter`."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["replace", "merge"] = "replace"
    cleanup: CleanupRules = Field(
        default=(),
        validation_alias=AliasChoices("cleanup", "cleanup_rules"),
    )
    validation_rules: FilterRules = Field(
        default=(),
        validation_alias=AliasChoices("validate", "filter_rules", "rules"),
    )

    def resolve(self, base: Filter | None = None) -> Filter:
        """Resolve the spec into the executable runtime filter object."""
        if self.mode == "replace":
            return Filter.define(cleanup_rules=self.cleanup, filter_rules=self.validation_rules)

        base_filter = Filter() if base is None else base
        return Filter.define(
            cleanup_rules=_merge_rules(base_filter.cleanup_rules, self.cleanup),
            filter_rules=_merge_rules(base_filter.filter_rules, self.validation_rules),
        )


def _merge_rules(base: tuple[RuleT, ...], updates: tuple[RuleT, ...]) -> tuple[RuleT, ...]:
    """Merge rules by stable rule name while preserving their visible order."""
    if not updates:
        return base

    updates_by_name = {rule_name(rule): rule for rule in updates}
    merged = [updates_by_name.pop(rule_name(rule), rule) for rule in base]
    merged.extend(updates_by_name.values())
    return tuple(merged)


def normalize_filter_frames(filters: Filter, *, sequence_length: int) -> Filter:
    """Resolve negative frame indices against a configured sequence length."""
    normalized_rules = tuple(
        _normalize_rule_frames(rule, sequence_length=sequence_length)
        for rule in filters.filter_rules
    )
    return Filter.define(
        cleanup_rules=filters.cleanup_rules,
        filter_rules=normalized_rules,
    )


def _normalize_rule_frames(rule: FilterRule, *, sequence_length: int) -> FilterRule:
    if isinstance(rule, RequireAgentFrames):
        return RequireAgentFrames(
            frames=_normalize_frame_set(rule.frames, sequence_length=sequence_length),
            max_invalid_agents=rule.max_invalid_agents,
            max_invalid_fraction=rule.max_invalid_fraction,
        )
    if isinstance(rule, RequireSceneFrames):
        return RequireSceneFrames(
            frames=_normalize_frame_set(rule.frames, sequence_length=sequence_length),
        )
    return rule


def _normalize_frame_set(
    frames: frozenset[int],
    *,
    sequence_length: int,
) -> frozenset[int]:
    normalized: set[int] = set()
    for frame in frames:
        resolved = sequence_length + frame if frame < 0 else frame
        if resolved < 0 or resolved >= sequence_length:
            msg = (
                f"Frame index {frame} is out of range for sequence length {sequence_length}. "
                f"Expected values in [{-sequence_length}, {sequence_length - 1}]."
            )
            raise ValueError(msg)
        normalized.add(resolved)
    return frozenset(normalized)
