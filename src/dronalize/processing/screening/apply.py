"""Application and diagnostics helpers for scene-screening rule sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from dronalize.core.polars_ops import normalize_group_by
from dronalize.processing.screening.agent import invalid_agent_tolerance_expr
from dronalize.processing.screening.base import (
    AgentCheckRuleBase,
    rule_name,
)
from dronalize.processing.screening.context import ScreeningContext

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.core.typing import DataFrameT
    from dronalize.processing.screening.base import CleanupRuleBase, Rule, SceneCheckRuleBase
    from dronalize.processing.screening.screen import Screen


_SCREENING_SCENE_PASSES = "_scene_passes"
AGENT_PASS_COLUMN = "_agent_passes"  # noqa: S105


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
        return cls(
            agent_pass=f"{prefix}_agent_passes",
            scene_pass=f"{prefix}_scene_passes",
        )


def screen_scene(
    data: DataFrameT,
    scene_screening: Screen | None,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str = "agent_category",
    *,
    mark_passed_agents: bool = False,
) -> DataFrameT:
    """Apply cleanup and screening rules to trajectory scenes."""
    if scene_screening is None:
        return data

    ctx = _build_context(
        group_by=group_by,
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
    )

    df = _apply_cleanup(data, scene_screening.cleanup_rules, ctx)
    df, scene_columns = _apply_scene_rules(df, scene_screening.scene_rules, ctx)
    df, agent_columns = _apply_agent_rules(
        df,
        scene_screening.agent_rules,
        ctx,
        include_passed_agent_ids=mark_passed_agents,
    )

    df = df.with_columns(
        _and_all([pl.col(name) for name in [*scene_columns, *agent_columns]]).alias(
            _SCREENING_SCENE_PASSES
        )
    )

    return _finalize_screened(
        df,
        diagnostic_columns=[*scene_columns, *agent_columns, _SCREENING_SCENE_PASSES],
        include_passed_agent_ids=mark_passed_agents,
    )


def _build_context(
    *,
    group_by: str | Sequence[str] | None,
    agent_id: str,
    frame_column: str,
    category_column: str,
) -> ScreeningContext:
    scene_window = list(normalize_group_by(group_by))
    agent_window = [*scene_window, agent_id] if scene_window else [agent_id]
    return ScreeningContext(
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
        scene_window=scene_window,
        agent_window=agent_window,
    )


def _apply_cleanup(
    data: DataFrameT,
    rules: tuple[CleanupRuleBase, ...],
    ctx: ScreeningContext,
) -> DataFrameT:
    return data.filter(_and_all([rule.expr(ctx) for rule in rules]))


def _apply_scene_rules(
    data: DataFrameT,
    rules: tuple[SceneCheckRuleBase, ...],
    ctx: ScreeningContext,
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
    data: DataFrameT,
    rule: AgentCheckRuleBase,
    ctx: ScreeningContext,
) -> tuple[DataFrameT, _RuleColumns]:
    cols = _RuleColumns.from_rule(rule)
    scope = ctx.selector_mask(rule.selector)
    scoped_agent_count = ctx.retained_agent_count(rule.selector)

    agent_pass = pl.when(scope).then(rule.expr(ctx)).otherwise(pl.lit(value=True))
    invalid_agents = (
        pl.col(ctx.agent_id).filter(scope & ~agent_pass).n_unique().over(ctx.scene_window)
    )
    invalid_fraction = (
        pl.when(scoped_agent_count > 0).then(invalid_agents / scoped_agent_count).otherwise(0.0)
    )
    scene_pass = (
        pl
        .when(scoped_agent_count > 0)
        .then(
            invalid_agent_tolerance_expr(
                rule.tolerance,
                invalid_agents=invalid_agents,
                invalid_fraction=invalid_fraction,
            )
        )
        .otherwise(pl.lit(value=True))
    )
    df = data.with_columns(
        agent_pass.alias(cols.agent_pass),
        scene_pass.alias(cols.scene_pass),
    )
    return df, cols


def _finalize_screened(
    data: DataFrameT,
    diagnostic_columns: list[str],
    *,
    include_passed_agent_ids: bool,
) -> DataFrameT:
    excluded = list(diagnostic_columns)
    if not include_passed_agent_ids:
        excluded.append(AGENT_PASS_COLUMN)

    screened = data.filter(pl.col(_SCREENING_SCENE_PASSES))
    return screened if not excluded else screened.select(pl.all().exclude(excluded))


def _agent_pass_expr(agent_pass_columns: list[str], ctx: ScreeningContext) -> pl.Expr:
    if not agent_pass_columns:
        return pl.lit(value=True)

    all_rules_passed = _and_all([pl.col(name) for name in agent_pass_columns])
    return ctx.over_agent_window(all_rules_passed.any())


def _and_all(exprs: list[pl.Expr]) -> pl.Expr:
    return pl.lit(value=True) if not exprs else pl.all_horizontal(*exprs)
