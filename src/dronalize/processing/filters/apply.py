from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import polars as pl

from dronalize._internal._polars_ops import normalize_group_by
from dronalize.processing.filters.context import FilterContext
from dronalize.processing.filters.rules.base import (
    AgentValidationRule,
    rule_name,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize._internal._typing import DataFrameT
    from dronalize.processing.filters.filter import Filter
    from dronalize.processing.filters.rules.base import CleanupRule, Rule, SceneValidationRule


_FILTER_SCENE_IS_VALID = "_filter_scene_is_valid"
FilterMode = Literal["filtered", "diagnose"]


@dataclass(slots=True, frozen=True)
class RuleDiagnostics:
    """Diagnostic column names emitted for a single rule."""

    agent_pass: str
    invalid_agents: str
    invalid_fraction: str
    scene_pass: str

    @classmethod
    def from_rule(cls, rule: Rule) -> RuleDiagnostics:
        """Build the diagnostic column names for one rule."""
        prefix = f"_filter_rule_{rule_name(rule)}"
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
        return [
            self.agent_pass,
            self.invalid_agents,
            self.invalid_fraction,
            self.scene_pass,
        ]


def filter_scene(
    data: DataFrameT,
    scene_filter: Filter | None,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = None,
    *,
    mode: FilterMode = "filtered",
) -> DataFrameT:
    """Apply cleanup and validation rules to trajectory scenes."""
    if scene_filter is None:
        return data

    diagnosed, diagnostic_columns = _diagnose_scene(
        data,
        scene_filter,
        group_by=group_by,
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
    )
    if mode == "diagnose":
        return diagnosed
    return _finalize_filtered(diagnosed, diagnostic_columns)


def _diagnose_scene(
    data: DataFrameT,
    scene_filter: Filter,
    *,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = None,
) -> tuple[DataFrameT, list[str]]:
    """Return cleaned data annotated with per-rule filter diagnostics."""
    ctx = _build_context(
        group_by=group_by,
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
    )
    cleaned = _apply_cleanup(data, scene_filter.cleanup_rules, ctx)
    diagnosed, diagnostic_columns, scene_pass_columns = _annotate_scene_rules(
        cleaned,
        scene_filter.scene_validation_rules,
        ctx,
    )
    return _annotate_agent_rules(
        diagnosed,
        scene_filter.agent_validation_rules,
        ctx,
        diagnostic_columns=diagnostic_columns,
        scene_pass_columns=scene_pass_columns,
    )


def _build_context(
    *,
    group_by: str | Sequence[str] | None,
    agent_id: str,
    frame_column: str,
    category_column: str | None,
) -> FilterContext:
    group_by_list = list(normalize_group_by(group_by))
    scene_window = group_by_list or pl.lit(1)
    agent_window = [*group_by_list, agent_id] if group_by_list else [agent_id]
    scene_start_frame = pl.col(frame_column).min().over(scene_window)
    return FilterContext(
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
        scene_window=scene_window,
        agent_window=agent_window,
        scene_start_frame=scene_start_frame,
    )


def _apply_cleanup(
    data: DataFrameT,
    rules: tuple[CleanupRule, ...],
    ctx: FilterContext,
) -> DataFrameT:
    return data.filter(_and_all([rule.expr(ctx) for rule in rules]))


def _annotate_scene_rules(
    data: DataFrameT,
    rules: tuple[SceneValidationRule, ...],
    ctx: FilterContext,
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
    rules: tuple[AgentValidationRule, ...],
    ctx: FilterContext,
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
        _and_all([pl.col(column) for column in scene_pass_columns]).alias(_FILTER_SCENE_IS_VALID),
    )
    diagnostic_columns.append(_FILTER_SCENE_IS_VALID)
    return diagnosed, diagnostic_columns


def _apply_scene_rule(
    data: DataFrameT,
    rule: SceneValidationRule,
    ctx: FilterContext,
) -> tuple[DataFrameT, RuleDiagnostics]:
    columns = RuleDiagnostics.from_rule(rule)
    diagnosed = data.with_columns(rule.expr(ctx).alias(columns.scene_pass))
    return diagnosed, columns


def _apply_agent_rule(
    data: DataFrameT,
    rule: AgentValidationRule,
    ctx: FilterContext,
) -> tuple[DataFrameT, RuleDiagnostics]:
    columns = RuleDiagnostics.from_rule(rule)
    data = data.with_columns(rule.expr(ctx).alias(columns.agent_pass))
    data = data.with_columns(
        pl
        .col(ctx.agent_id)
        .filter(~pl.col(columns.agent_pass))
        .n_unique()
        .over(ctx.scene_window)
        .alias(columns.invalid_agents),
    )
    data = data.with_columns(
        pl
        .when(ctx.retained_agent_count() > 0)
        .then(pl.col(columns.invalid_agents) / ctx.retained_agent_count())
        .otherwise(1.0)
        .alias(columns.invalid_fraction),
    )
    data = data.with_columns(
        pl.all_horizontal(
            pl.col(columns.invalid_agents) <= rule.max_invalid_agents,
            pl.col(columns.invalid_fraction) <= rule.max_invalid_fraction,
        ).alias(columns.scene_pass),
    )
    return data, columns


def _finalize_filtered(data: DataFrameT, diagnostic_columns: list[str]) -> DataFrameT:
    filtered = data.filter(pl.col(_FILTER_SCENE_IS_VALID))
    if len(diagnostic_columns) == 0:
        return filtered
    return filtered.select(pl.all().exclude(diagnostic_columns))


def _and_all(exprs: list[pl.Expr]) -> pl.Expr:
    if len(exprs) == 0:
        return pl.lit(value=True)
    return pl.all_horizontal(*exprs)
