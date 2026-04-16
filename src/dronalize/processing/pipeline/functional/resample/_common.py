"""Shared models and planning helpers for temporal resampling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING, Final, Literal

import polars as pl

from dronalize.core.errors import LoaderConfigError
from dronalize.core.polars_ops import normalize_group_by

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.core.typing import DataFrameT

SEGMENT_COLUMN: Final = "_resample_segment"

CoordinateColumns = tuple[str, ...]
DerivativeColumns = tuple[str, ...]
EmittedDerivative = Literal["velocity", "acceleration"]


class ResampleMethod(str, Enum):
    """Interpolation strategy used during resampling."""

    LINEAR = "linear"
    CUBIC = "cubic"
    PCHIP = "pchip"

    def supports_derivatives(self) -> bool:
        """Return whether the method can emit derivative columns."""
        return self is not ResampleMethod.LINEAR


def _velocity_column_name(coordinate: str) -> str:
    return f"v{coordinate}" if len(coordinate) == 1 else f"velocity_{coordinate}"


def _acceleration_column_name(coordinate: str) -> str:
    return f"a{coordinate}" if len(coordinate) == 1 else f"acceleration_{coordinate}"


@dataclass(frozen=True, slots=True)
class ResampleSpec:
    """Validated specification for temporal resampling."""

    up: int = 1
    down: int = 1
    method: ResampleMethod = ResampleMethod.LINEAR
    coordinates: CoordinateColumns = ("x", "y")
    emit_velocity: bool = False
    emit_acceleration: bool = False
    max_gap: int = 1
    sample_time: float = 1.0

    def __post_init__(self) -> None:
        if self.up <= 0 or self.down <= 0:
            msg = "Resampling factors 'up' and 'down' must be positive."
            raise LoaderConfigError(msg)
        if self.max_gap <= 0:
            msg = "Resampling 'max_gap' must be positive."
            raise LoaderConfigError(msg)
        if self.sample_time <= 0.0:
            msg = "Resampling 'sample_time' must be positive."
            raise LoaderConfigError(msg)

        simplified_up, simplified_down = Fraction(self.up, self.down).as_integer_ratio()
        object.__setattr__(self, "up", simplified_up)
        object.__setattr__(self, "down", simplified_down)

        if self.method is ResampleMethod.LINEAR and (self.emit_acceleration or self.emit_velocity):
            msg = "Linear resampling does not support emitting derivatives."
            raise LoaderConfigError(msg)

    @classmethod
    def cubic(cls, up: int = 1, down: int = 1) -> ResampleSpec:
        """Return a spec for cubic spline resampling."""
        return cls(up=up, down=down, method=ResampleMethod.CUBIC)

    @classmethod
    def pchip(cls, up: int = 1, down: int = 1) -> ResampleSpec:
        """Return a spec for PCHIP resampling."""
        return cls(up=up, down=down, method=ResampleMethod.PCHIP)

    @classmethod
    def linear(cls, up: int = 1, down: int = 1) -> ResampleSpec:
        """Return a spec for linear resampling."""
        return cls(up=up, down=down, method=ResampleMethod.LINEAR)

    def _with_update(self, **kwargs: object) -> ResampleSpec:
        return replace(self, **kwargs)

    def with_sample_time(self, sample_time: float) -> ResampleSpec:
        """Return a copy with a different sampling interval."""
        return self._with_update(sample_time=sample_time)

    def with_emit(
        self, *, acceleration: bool | None = None, velocity: bool | None = None
    ) -> ResampleSpec:
        """Return a copy with different derivative emission settings."""
        emit_velocity = self.emit_velocity if velocity is None else velocity
        emit_acceleration = self.emit_acceleration if acceleration is None else acceleration
        return self._with_update(emit_velocity=emit_velocity, emit_acceleration=emit_acceleration)

    @property
    def no_resampling(self) -> bool:
        """Return whether the specification keeps the original sampling rate."""
        return self.up == 1 and self.down == 1


@dataclass(frozen=True, slots=True)
class ResamplePlan:
    """Execution-ready resampling plan derived from a :class:`ResampleSpec`."""

    frame_column: str
    group_by: tuple[str, ...]
    segment_keys: tuple[str, ...]
    coordinates: CoordinateColumns
    velocity_columns: DerivativeColumns
    acceleration_columns: DerivativeColumns
    emitted_columns: tuple[str, ...]

    @property
    def emit_velocity(self) -> bool:
        return len(self.velocity_columns) > 0

    @property
    def emit_acceleration(self) -> bool:
        return len(self.acceleration_columns) > 0

    @property
    def packed_columns(self) -> tuple[str, ...]:
        return (self.frame_column, *self.coordinates)


def resolve_request(
    spec: ResampleSpec, *, frame_column: str, group_by: str | Sequence[str] | None
) -> ResamplePlan:
    """Normalize a resampling specification into an execution plan."""
    group_columns = normalize_group_by(group_by)
    velocity_columns = (
        tuple(_velocity_column_name(coordinate) for coordinate in spec.coordinates)
        if spec.emit_velocity
        else ()
    )
    acceleration_columns = (
        tuple(_acceleration_column_name(coordinate) for coordinate in spec.coordinates)
        if spec.emit_acceleration
        else ()
    )
    emitted_columns = (frame_column, *spec.coordinates, *velocity_columns, *acceleration_columns)
    return ResamplePlan(
        frame_column=frame_column,
        group_by=group_columns,
        segment_keys=(*group_columns, SEGMENT_COLUMN),
        coordinates=spec.coordinates,
        velocity_columns=velocity_columns,
        acceleration_columns=acceleration_columns,
        emitted_columns=emitted_columns,
    )


def segment_data(
    data: DataFrameT, *, frame_column: str, group_by: Sequence[str], max_gap: int
) -> DataFrameT:
    """Annotate contiguous trajectory segments separated by gaps larger than `max_gap`."""
    data = data.sort([*group_by, frame_column])
    expr = (pl.col(frame_column).diff() > max_gap).fill_null(value=False).cum_sum()
    if group_by:
        expr = expr.over(group_by)

    return data.with_columns(expr.alias(SEGMENT_COLUMN))


def packed_struct_expression(plan: ResamplePlan) -> pl.Expr:
    """Return a struct expression containing the columns resampling operates on."""
    return pl.struct(pl.col(column) for column in plan.packed_columns)


def not_packed_columns(plan: ResamplePlan) -> pl.Expr:
    """Return an expression selecting columns carried through unchanged."""
    excluded = [*plan.group_by, *plan.emitted_columns, SEGMENT_COLUMN]
    return pl.all().exclude(excluded)
