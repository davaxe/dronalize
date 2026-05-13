"""Screen containers, declarative configs, and application helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Generic, TypeVar

import polars as pl
from pydantic import BaseModel, BeforeValidator, TypeAdapter
from typing_extensions import override

from dronalize.core.functional.basic import normalize_group_by
from dronalize.processing.screening.agent import AgentCheckRule, invalid_agent_tolerance_expr
from dronalize.processing.screening.base import (
    AgentCheckRuleBase,
    Rule,
    ScreeningContext,
    rule_name,
)
from dronalize.processing.screening.cleanup import CleanupRule, PruneByRule
from dronalize.processing.screening.scene import SceneCheckRule

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dronalize.config.models import ScreeningConfig
    from dronalize.core.typing import DataFrameT
    from dronalize.processing.columns import TrajectoryColumns
    from dronalize.processing.screening.base import CleanupRuleBase, SceneCheckRuleBase


_RuleT = TypeVar("_RuleT", bound=Rule)
_RuleSpecT = TypeVar("_RuleSpecT", bound=BaseModel)
AgentCheckSpecs = Annotated[tuple[AgentCheckRule, ...], BeforeValidator(tuple)]
CleanupSpecs = Annotated[tuple[CleanupRule, ...], BeforeValidator(tuple)]
SceneCheckSpecs = Annotated[tuple[SceneCheckRule, ...], BeforeValidator(tuple)]
SCENE_PASS_COLUMN: Final[str] = "_scene_passes"  # noqa: S105
AGENT_PASS_COLUMN: Final[str] = "_agent_passes"  # noqa: S105


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


@dataclass(slots=True, frozen=True)
class _RuleColumns:
    """Diagnostic column names emitted for a single screening rule."""

    agent_pass: str
    scene_pass: str

    @classmethod
    def from_rule(cls, rule: Rule) -> _RuleColumns:
        """Return stable rule column name for a given rule."""
        prefix = f"_screening_rule_{rule_name(rule)}"
        if isinstance(rule, AgentCheckRuleBase):
            prefix = "_agent" + prefix
        return cls(agent_pass=f"{prefix}_agent_passes", scene_pass=f"{prefix}_scene_passes")


def screen_scene(
    data: DataFrameT,
    scene_screening: Screen | None,
    columns: TrajectoryColumns,
    scene_group_by: str | Sequence[str] | None = None,
    *,
    mark_passed_agents: bool = False,
    retain_scene_passes: bool = False,
) -> DataFrameT:
    """Apply cleanup and screening rules to trajectory scenes."""
    if scene_screening is None:
        return data

    ctx = _build_context(columns=columns, scene_group_by=scene_group_by)
    df = _apply_cleanup(data, scene_screening.cleanup_rules, ctx)
    df, scene_columns = _apply_scene_rules(df, scene_screening.scene_rules, ctx)
    df, agent_columns = _apply_agent_rules(
        df, scene_screening.agent_rules, ctx, include_passed_agent_ids=mark_passed_agents
    )
    df = df.with_columns(
        _and_all([pl.col(name) for name in [*scene_columns, *agent_columns]]).alias(
            SCENE_PASS_COLUMN
        )
    )
    return _finalize_screened(
        df,
        diagnostic_columns=[*scene_columns, *agent_columns, SCENE_PASS_COLUMN],
        include_passed_agent_ids=mark_passed_agents,
        retain_scene_passes=retain_scene_passes,
    )


def _build_context(
    *, columns: TrajectoryColumns, scene_group_by: str | Sequence[str] | None
) -> ScreeningContext:
    scene_window = list(normalize_group_by(scene_group_by))
    agent_window = [*scene_window, columns.agent_id] if scene_window else [columns.agent_id]
    return ScreeningContext(
        columns=columns, scene_window=tuple(scene_window), agent_window=tuple(agent_window)
    )


def _apply_cleanup(
    data: DataFrameT, rules: tuple[CleanupRuleBase, ...], ctx: ScreeningContext
) -> DataFrameT:
    return data.filter(_and_all([rule.expr(ctx) for rule in rules]))


def _apply_scene_rules(
    data: DataFrameT, rules: tuple[SceneCheckRuleBase, ...], ctx: ScreeningContext
) -> tuple[DataFrameT, list[str]]:
    df = data
    scene_passes: list[str] = []
    for rule in rules:
        cols = _RuleColumns.from_rule(rule)
        df = df.with_columns(rule.expr(ctx).alias(cols.scene_pass))
        scene_passes.append(cols.scene_pass)

    return df, scene_passes


def _apply_agent_rules(
    data: DataFrameT,
    rules: tuple[AgentCheckRuleBase, ...],
    ctx: ScreeningContext,
    *,
    include_passed_agent_ids: bool,
) -> tuple[DataFrameT, list[str]]:
    df = data
    scene_passes: list[str] = []
    agent_passes: list[str] = []

    for rule in rules:
        df, cols = _apply_agent_rule(df, rule, ctx)
        scene_passes.append(cols.scene_pass)
        agent_passes.append(cols.agent_pass)

    if include_passed_agent_ids:
        df = df.with_columns(_agent_pass_expr(agent_passes, ctx).alias(AGENT_PASS_COLUMN))

    df = df.drop(agent_passes)
    return df, scene_passes


def _apply_agent_rule(
    data: DataFrameT, rule: AgentCheckRuleBase, ctx: ScreeningContext
) -> tuple[DataFrameT, _RuleColumns]:
    cols = _RuleColumns.from_rule(rule)
    scope = ctx.selector_mask(rule.selector)
    scoped_agent_count = ctx.retained_agent_count(rule.selector)

    agent_pass = pl.when(scope).then(rule.expr(ctx)).otherwise(pl.lit(value=True))
    invalid_agents = ctx.over_scene_window(
        pl.col(ctx.columns.agent_id).filter(scope & ~agent_pass).n_unique()
    )
    invalid_fraction = (
        pl.when(scoped_agent_count > 0).then(invalid_agents / scoped_agent_count).otherwise(0.0)
    )
    scene_pass = (
        pl
        .when(scoped_agent_count > 0)
        .then(
            invalid_agent_tolerance_expr(
                rule.tolerance, invalid_agents=invalid_agents, invalid_fraction=invalid_fraction
            )
        )
        .otherwise(pl.lit(value=True))
    )
    df = data.with_columns(agent_pass.alias(cols.agent_pass), scene_pass.alias(cols.scene_pass))
    return df, cols


def _finalize_screened(
    data: DataFrameT,
    diagnostic_columns: list[str],
    *,
    include_passed_agent_ids: bool,
    retain_scene_passes: bool,
) -> DataFrameT:
    excluded = list(diagnostic_columns)
    if not include_passed_agent_ids:
        excluded.append(AGENT_PASS_COLUMN)
    if retain_scene_passes:
        excluded.remove(SCENE_PASS_COLUMN)
        return data if not excluded else data.select(pl.all().exclude(excluded))

    screened = data.filter(pl.col(SCENE_PASS_COLUMN))
    return screened if not excluded else screened.select(pl.all().exclude(excluded))


def _agent_pass_expr(agent_pass_columns: list[str], ctx: ScreeningContext) -> pl.Expr:
    if not agent_pass_columns:
        return pl.lit(value=True)

    all_rules_passed = _and_all([pl.col(name) for name in agent_pass_columns])
    return ctx.over_agent_window(all_rules_passed.any())


def _and_all(exprs: list[pl.Expr]) -> pl.Expr:
    return pl.lit(value=True) if not exprs else pl.all_horizontal(*exprs)
