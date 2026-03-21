from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count
from typing import Final

import polars as pl

from dronalize.pipeline.functional.basic import yaw_from_vel
from dronalize.pipeline.functional.derivative import derivative
from dronalize.scene._schema import SceneField

_POSITION_FIELDS: Final[SceneField] = SceneField.X | SceneField.Y
_VELOCITY_FIELDS: Final[SceneField] = SceneField.VX | SceneField.VY
_ACCELERATION_FIELDS: Final[SceneField] = SceneField.AX | SceneField.AY
_YAW_FIELDS: Final[SceneField] = SceneField.YAW
_KINEMATIC_FIELDS: Final[SceneField] = _VELOCITY_FIELDS | _ACCELERATION_FIELDS

_DERIVATIVE_RENAME: Final[dict[int, list[str]]] = {1: ["vx", "vy"], 2: ["ax", "ay"]}
_TMP_YAW_VELOCITY: Final[tuple[str, str]] = ("__scene_tmp_vx", "__scene_tmp_vy")


@dataclass(slots=True, frozen=True)
class ConversionContext:
    """Runtime information needed to derive scene fields."""

    sample_time: float | None
    group_by: str | Sequence[str] | None = "id"


DerivationApply = Callable[[pl.LazyFrame, ConversionContext], pl.LazyFrame]


@dataclass(slots=True, frozen=True)
class DerivationRule:
    """A rule that derives one or more semantic scene fields."""

    name: str
    """Unique name for this rule."""
    requires: SceneField
    """Fields required to apply this rule."""
    outputs: SceneField
    """Fields added by applying this rule."""
    cost: int
    """Relative cost of applying this rule, used for derivation planning."""
    apply: DerivationApply
    """Function that applies the rule to a dataframe."""
    needs_sample_time: bool = False
    """Flag to indicate if the rule requires sample time."""

    def is_applicable(
        self,
        available: SceneField,
        context: ConversionContext,
    ) -> bool:
        """Return True if the rule can be applied in the current state."""
        has_inputs = (available & self.requires) == self.requires
        adds_new_fields = (available & self.outputs) != self.outputs
        has_required_time = not self.needs_sample_time or context.sample_time is not None
        return has_inputs and adds_new_fields and has_required_time


def apply_derivation_plan(
    data: pl.LazyFrame,
    plan: Iterable[DerivationRule],
    context: ConversionContext,
) -> pl.LazyFrame:
    """Apply a derivation plan in order."""
    for rule in plan:
        data = rule.apply(data, context)
    return data


@functools.cache
def plan_derivations(
    available_fields: SceneField,
    required_fields: SceneField,
    context: ConversionContext,
) -> tuple[DerivationRule, ...] | None:
    """Find the optimal plan to derive the required fields.

    Uses Dijkstra's algorithm to find the lowest-cost sequence of derivation
    rules that can produce the required fields from the available fields, given
    the constraints of the conversion context.

    Parameters
    ----------
    available_fields : SceneField
        Bitmask of fields currently available.
    required_fields : SceneField
        Bitmask of fields that need to be derived.
    context : ConversionContext
        Additional information that may affect rule applicability, such as
        sample time.

    """
    start = available_fields
    goal = required_fields

    if (start & goal) == goal:
        return ()

    best_cost: dict[SceneField, int] = {start: 0}
    came_from: dict[SceneField, tuple[SceneField, DerivationRule]] = {}
    queue: list[tuple[int, int, SceneField]] = []
    tie_breaker = count()
    heappush(queue, (0, next(tie_breaker), start))

    while queue:
        cost, _, state = heappop(queue)
        if cost > best_cost.get(state, float("inf")):
            continue
        if (state & goal) == goal:
            plan: list[DerivationRule] = []
            curr_state = state
            while curr_state in came_from:
                curr_state, rule = came_from[curr_state]
                plan.append(rule)
            return tuple(reversed(plan))

        for rule in DERIVATION_RULES:
            if not rule.is_applicable(state, context):
                continue
            next_state = state | rule.outputs
            next_cost = cost + rule.cost
            if next_cost < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = next_cost
                came_from[next_state] = (state, rule)
                heappush(queue, (next_cost, next(tie_breaker), next_state))

    return None


def can_plan_with_sample_time(
    available_fields: SceneField,
    required_fields: SceneField,
    context: ConversionContext,
) -> bool:
    """Return whether a derivation plan exists if sample time is available."""
    trial_context: ConversionContext = context
    if context.sample_time is None:
        trial_context = replace(context, sample_time=1.0)

    return plan_derivations(available_fields, required_fields, trial_context) is not None


def requires_sample_time(
    missing_fields: SceneField,
    available_fields: SceneField,
    context: ConversionContext,
) -> bool:
    """Return True if the missing fields become derivable only when sample_time is provided."""
    missing = missing_fields
    available = available_fields
    if (context.sample_time is not None or not missing) or not (missing & _KINEMATIC_FIELDS):
        return False

    has_source_for_kinematics = ((available & _POSITION_FIELDS) == _POSITION_FIELDS) or (
        (available & _VELOCITY_FIELDS) == _VELOCITY_FIELDS
    )
    if not has_source_for_kinematics:
        return False

    return can_plan_with_sample_time(available, available | missing, context)


def _require_sample_time(sample_time: float | None) -> float:
    if sample_time is None:
        msg = "Scene schema conversion requires sample_time to derive kinematics."
        raise ValueError(msg)
    return sample_time


def _apply_derivative(
    data: pl.LazyFrame,
    context: ConversionContext,
    *,
    x_col: str,
    y_col: str,
    order: int,
    rename: dict[int, list[str]],
    include_intermediate: bool = False,
    dt: float | None = None,
) -> pl.LazyFrame:
    """Shared helper for derivative-based field derivation."""
    return derivative(
        data,
        x_col,
        y_col,
        dt=_require_sample_time(context.sample_time) if dt is None else dt,
        n=order,
        include_intermediate=include_intermediate,
        group_by=context.group_by,
        derivative_rename=rename,
    )


def _velocity_from_position(data: pl.LazyFrame, context: ConversionContext) -> pl.LazyFrame:
    return _apply_derivative(
        data,
        context,
        x_col="x",
        y_col="y",
        order=1,
        rename={1: ["vx", "vy"]},
    )


def _acceleration_from_velocity(data: pl.LazyFrame, context: ConversionContext) -> pl.LazyFrame:
    return _apply_derivative(
        data,
        context,
        x_col="vx",
        y_col="vy",
        order=1,
        rename={1: ["ax", "ay"]},
    )


def _acceleration_from_position(data: pl.LazyFrame, context: ConversionContext) -> pl.LazyFrame:
    return _apply_derivative(
        data,
        context,
        x_col="x",
        y_col="y",
        order=2,
        rename={2: ["ax", "ay"]},
    )


def _kinematics_from_position(data: pl.LazyFrame, context: ConversionContext) -> pl.LazyFrame:
    return _apply_derivative(
        data,
        context,
        x_col="x",
        y_col="y",
        order=2,
        rename=_DERIVATIVE_RENAME,
        include_intermediate=True,
    )


def _yaw_from_velocity(data: pl.LazyFrame, context: ConversionContext) -> pl.LazyFrame:
    _ = context
    return yaw_from_vel(data, "vx", "vy", "yaw")


def _yaw_from_position(data: pl.LazyFrame, context: ConversionContext) -> pl.LazyFrame:
    data = _apply_derivative(
        data,
        context,
        x_col="x",
        y_col="y",
        order=1,
        rename={1: list(_TMP_YAW_VELOCITY)},
        dt=1.0,
    )
    return yaw_from_vel(data, _TMP_YAW_VELOCITY[0], _TMP_YAW_VELOCITY[1], "yaw").drop(
        list(_TMP_YAW_VELOCITY)
    )


DERIVATION_RULES: Final[tuple[DerivationRule, ...]] = (
    DerivationRule(
        name="velocity_from_position",
        requires=_POSITION_FIELDS,
        outputs=_VELOCITY_FIELDS,
        cost=10,
        apply=_velocity_from_position,
        needs_sample_time=True,
    ),
    DerivationRule(
        name="acceleration_from_velocity",
        requires=_VELOCITY_FIELDS,
        outputs=_ACCELERATION_FIELDS,
        cost=10,
        apply=_acceleration_from_velocity,
        needs_sample_time=True,
    ),
    DerivationRule(
        name="acceleration_from_position",
        requires=_POSITION_FIELDS,
        outputs=_ACCELERATION_FIELDS,
        cost=13,
        apply=_acceleration_from_position,
        needs_sample_time=True,
    ),
    DerivationRule(
        name="kinematics_from_position",
        requires=_POSITION_FIELDS,
        outputs=_KINEMATIC_FIELDS,
        cost=14,
        apply=_kinematics_from_position,
        needs_sample_time=True,
    ),
    DerivationRule(
        name="yaw_from_velocity",
        requires=_VELOCITY_FIELDS,
        outputs=_YAW_FIELDS,
        cost=5,
        apply=_yaw_from_velocity,
    ),
    DerivationRule(
        name="yaw_from_position",
        requires=_POSITION_FIELDS,
        outputs=_YAW_FIELDS,
        cost=8,
        apply=_yaw_from_position,
    ),
)
