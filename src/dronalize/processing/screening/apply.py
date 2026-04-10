"""Application and diagnostics helpers for scene-screening rule sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import polars as pl

from dronalize.core.polars_ops import normalize_group_by
from dronalize.processing.screening.agent import invalid_agent_tolerance_expr
from dronalize.processing.screening.base import AgentCheckRuleBase, rule_name
from dronalize.processing.screening.context import ScreeningContext

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.core.typing import DataFrameT
    from dronalize.processing.screening.base import CleanupRuleBase, Rule, SceneCheckRuleBase
    from dronalize.processing.screening.screen import Screen


_SCREENING_SCENE_PASSES = "_screening_scene_passes"
ScreeningMode = Literal["screened", "diagnose"]


@dataclass(slots=True, frozen=True)
class RuleDiagnostics:
    """Diagnostic column names emitted for a single screening rule."""

    agent_pass: str
    invalid_agents: str
    invalid_fraction: str
    scene_pass: str

    @classmethod
    def from_rule(cls, rule: Rule) -> RuleDiagnostics:
        """Build the diagnostic column names for one rule."""
        prefix = f"_screening_rule_{rule_name(rule)}"
        return cls(
            agent_pass=f"{prefix}_agent_passes",
            invalid_agents=f"{prefix}_invalid_agents",
            invalid_fraction=f"{prefix}_invalid_fraction",
            scene_pass=f"{prefix}_scene_passes",
        )

    def scene_names(self) -> list[str]:
        """Return the columns created for a scene-scoped rule."""
        return [self.scene_pass]

    def agent_names(self) -> list[str]:
        """Return the columns created for an agent-scoped rule."""
        return [self.agent_pass, self.invalid_agents, self.invalid_fraction, self.scene_pass]


def screen_scene(
    data: DataFrameT,
    scene_screening: Screen | None,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str = "agent_category",
    *,
    mode: ScreeningMode = "screened",
) -> DataFrameT:
    """Apply cleanup and check rules to trajectory scenes."""
    if scene_screening is None:
        return data

    diagnosed, diagnostic_columns = _diagnose_scene(
        data,
        scene_screening,
        group_by=group_by,
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
    )
    if mode == "diagnose":
        return diagnosed
    return _finalize_screened(diagnosed, diagnostic_columns)


def _diagnose_scene(
    data: DataFrameT,
    scene_screening: Screen,
    *,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str = "agent_category",
) -> tuple[DataFrameT, list[str]]:
    """Return cleaned data annotated with per-rule screening diagnostics."""
    ctx = _build_context(
        group_by=group_by,
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
    )
    cleaned = _apply_cleanup(data, scene_screening.cleanup_rules, ctx)
    diagnosed, diagnostic_columns, scene_pass_columns = _annotate_scene_rules(
        cleaned, scene_screening.scene_rules, ctx
    )
    return _annotate_agent_rules(
        diagnosed,
        scene_screening.agent_rules,
        ctx,
        diagnostic_columns=diagnostic_columns,
        scene_pass_columns=scene_pass_columns,
    )


def _build_context(
    *, group_by: str | Sequence[str] | None, agent_id: str, frame_column: str, category_column: str
) -> ScreeningContext:
    group_by_list = list(normalize_group_by(group_by))
    scene_window = group_by_list
    agent_window = [*group_by_list, agent_id] if group_by_list else [agent_id]
    return ScreeningContext(
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
        scene_window=scene_window,
        agent_window=agent_window,
    )


def _apply_cleanup(
    data: DataFrameT, rules: tuple[CleanupRuleBase, ...], ctx: ScreeningContext
) -> DataFrameT:
    return data.filter(_and_all([rule.expr(ctx) for rule in rules]))


def _annotate_scene_rules(
    data: DataFrameT, rules: tuple[SceneCheckRuleBase, ...], ctx: ScreeningContext
) -> tuple[DataFrameT, list[str], list[str]]:
    diagnostic_columns: list[str] = []
    scene_pass_columns: list[str] = []
    diagnosed = data

    for rule in rules:
        diagnosed, columns = _apply_scene_rule(diagnosed, rule, ctx)
        scene_pass_columns.append(columns.scene_pass)
        diagnostic_columns.extend(columns.scene_names())

    return diagnosed, diagnostic_columns, scene_pass_columns


def _annotate_agent_rules(
    data: DataFrameT,
    rules: tuple[AgentCheckRuleBase, ...],
    ctx: ScreeningContext,
    *,
    diagnostic_columns: list[str],
    scene_pass_columns: list[str],
) -> tuple[DataFrameT, list[str]]:
    diagnosed = data

    for rule in rules:
        diagnosed, columns = _apply_agent_rule(diagnosed, rule, ctx)
        scene_pass_columns.append(columns.scene_pass)
        diagnostic_columns.extend(columns.agent_names())

    diagnosed = diagnosed.with_columns(
        _and_all([pl.col(column) for column in scene_pass_columns]).alias(_SCREENING_SCENE_PASSES)
    )
    diagnostic_columns.append(_SCREENING_SCENE_PASSES)
    return diagnosed, diagnostic_columns


def _apply_scene_rule(
    data: DataFrameT, rule: SceneCheckRuleBase, ctx: ScreeningContext
) -> tuple[DataFrameT, RuleDiagnostics]:
    columns = RuleDiagnostics.from_rule(rule)
    diagnosed = data.with_columns(rule.expr(ctx).alias(columns.scene_pass))
    return diagnosed, columns


def _apply_agent_rule(
    data: DataFrameT, rule: AgentCheckRuleBase, ctx: ScreeningContext
) -> tuple[DataFrameT, RuleDiagnostics]:
    columns = RuleDiagnostics.from_rule(rule)
    scope = ctx.selector_mask(rule.selector)
    scoped_agent_count = ctx.retained_agent_count(rule.selector)
    data = data.with_columns(
        pl.when(scope).then(rule.expr(ctx)).otherwise(statement=True).alias(columns.agent_pass)
    )
    data = data.with_columns(
        pl
        .col(ctx.agent_id)
        .filter(scope & ~pl.col(columns.agent_pass))
        .n_unique()
        .over(ctx.scene_window)
        .alias(columns.invalid_agents)
    )
    data = data.with_columns(
        pl
        .when(scoped_agent_count > 0)
        .then(pl.col(columns.invalid_agents) / scoped_agent_count)
        .otherwise(0.0)
        .alias(columns.invalid_fraction)
    )
    data = data.with_columns(
        pl
        .when(scoped_agent_count > 0)
        .then(
            invalid_agent_tolerance_expr(
                rule.tolerance,
                invalid_agents=pl.col(columns.invalid_agents),
                invalid_fraction=pl.col(columns.invalid_fraction),
            )
        )
        .otherwise(statement=True)
        .alias(columns.scene_pass)
    )
    return data, columns


def _finalize_screened(data: DataFrameT, diagnostic_columns: list[str]) -> DataFrameT:
    screened = data.filter(pl.col(_SCREENING_SCENE_PASSES))
    if len(diagnostic_columns) == 0:
        return screened
    return screened.select(pl.all().exclude(diagnostic_columns))


def _and_all(exprs: list[pl.Expr]) -> pl.Expr:
    if len(exprs) == 0:
        return pl.lit(value=True)
    return pl.all_horizontal(*exprs)
