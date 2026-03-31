from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from itertools import chain
from typing import TYPE_CHECKING, Annotated, ClassVar, Final

import polars as pl
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self, override

from dronalize._internal.polars_ops import normalize_group_by

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize._internal.typing import DataFrameT

SEGMENT_COLUMN: Final = "_resample_segment"

ColumnOrder = dict[str, None]
DerivativeOrderMap = dict[int, ColumnOrder]


class ResampleMethod(str, Enum):
    """Interpolation strategy used during resampling."""

    LINEAR = "linear"
    CUBIC = "cubic"
    PCHIP = "pchip"
    HERMITE = "hermite"

    def support_derivatives(self) -> bool:
        """Whether the method supports input or output derivatives."""
        return self in {ResampleMethod.CUBIC, ResampleMethod.PCHIP, ResampleMethod.HERMITE}


def _map_alias(v: str) -> str | ResampleMethod:
    alias_map = {
        "cubic_spline": ResampleMethod.CUBIC,
        "pchip_interpolation": ResampleMethod.PCHIP,
        "hermite_spline": ResampleMethod.HERMITE,
        "fast": ResampleMethod.LINEAR,
        "spline": ResampleMethod.CUBIC,
    }
    return alias_map.get(v, v)


AliasedResampling = Annotated[ResampleMethod, BeforeValidator(_map_alias)]


class ResampleSpec(BaseModel):
    """Validated specification for temporal resampling."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    up: int = Field(default=1, gt=0)
    down: int = Field(default=1, gt=0)
    method: AliasedResampling = Field(default=ResampleMethod.LINEAR)
    position_columns: ColumnOrder = Field(default_factory=lambda: dict.fromkeys(("x", "y")))
    input_derivatives: DerivativeOrderMap = Field(default_factory=dict)
    output_derivatives: DerivativeOrderMap = Field(default_factory=dict)
    max_gap: int = Field(default=1, gt=0)
    sort: bool = Field(default=True)
    sample_time: float = Field(default=1.0, gt=0.0)

    @classmethod
    def cubic(cls, up: int = 1, down: int = 1) -> ResampleSpec:
        """Return a spec for cubic spline resampling."""
        return cls(up=up, down=down, method=ResampleMethod.CUBIC)

    @classmethod
    def pchip(cls, up: int = 1, down: int = 1) -> ResampleSpec:
        """Return a spec for PCHIP resampling."""
        return cls(up=up, down=down, method=ResampleMethod.PCHIP)

    @classmethod
    def hermite(cls, up: int = 1, down: int = 1) -> ResampleSpec:
        """Return a spec for Hermite spline resampling."""
        return cls(up=up, down=down, method=ResampleMethod.HERMITE)

    @classmethod
    def linear(cls, up: int = 1, down: int = 1) -> ResampleSpec:
        """Return a spec for linear resampling."""
        return cls(up=up, down=down, method=ResampleMethod.LINEAR)

    def _with_update(self, **kwargs: object) -> ResampleSpec:
        return self.model_copy(update=kwargs)

    def with_input_derivative(self, order: int, *columns: str) -> ResampleSpec:
        """Return a new spec with the given input derivative order and columns."""
        return self._with_update(
            input_derivatives={**self.input_derivatives, order: dict.fromkeys(columns)}
        )

    def with_output_derivative(self, order: int, *columns: str) -> ResampleSpec:
        """Return a new spec with the given output derivative order and columns."""
        return self._with_update(
            output_derivatives={**self.output_derivatives, order: dict.fromkeys(columns)}
        )

    def with_sample_time(self, sample_time: float) -> ResampleSpec:
        """Return a new spec with the given sample time."""
        return self._with_update(sample_time=sample_time)

    def with_default_output_deivative(
        self, *, first: bool = True, second: bool = True
    ) -> ResampleSpec:
        """Return a spec wit default added output derivatives.

        By default adds:
         - First-order output derivatives "vx", "vy"
         - Second-order output derivatives "ax", "ay"

        """
        new = self
        if first:
            new = self.with_output_derivative(1, "vx", "vy")
        if second:
            new = self.with_output_derivative(2, "ax", "ay")
        return new

    @field_validator("position_columns", mode="before")
    @classmethod
    def _validate_position_columns(cls, value: ColumnOrder | Sequence[str]) -> ColumnOrder:
        cols = dict.fromkeys(value.keys() if isinstance(value, dict) else value)
        if not cols:
            msg = "position_columns must contain at least one column."
            raise ValueError(msg)
        return cols

    @field_validator("input_derivatives", "output_derivatives", mode="before")
    @classmethod
    def _normalize_derivatives(
        cls, value: dict[int, ColumnOrder | Sequence[str]] | None
    ) -> DerivativeOrderMap:
        return {
            int(order): dict.fromkeys(cols.keys() if isinstance(cols, dict) else cols)
            for order, cols in (value or {}).items()
        }

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        # Simplify fraction
        simplified_up, simplified_down = Fraction(self.up, self.down).as_integer_ratio()
        object.__setattr__(self, "up", simplified_up)
        object.__setattr__(self, "down", simplified_down)
        pos_len = len(self.position_columns)

        # Validate derivative shapes
        for name, mapping in (
            ("input", self.input_derivatives),
            ("output", self.output_derivatives),
        ):
            for order, cols in mapping.items():
                if order <= 0:
                    msg = f"{name}_derivatives keys must be positive derivative orders."
                    raise ValueError(msg)
                if len(cols) != pos_len:
                    msg = f"{name}_derivatives[{order}] must match position_columns length."
                    raise ValueError(msg)

        # Validate method rules
        has_in, has_out = bool(self.input_derivatives), bool(self.output_derivatives)
        if self.method is ResampleMethod.LINEAR and (has_in or has_out):
            msg = "Linear resampling does not support derivative inputs or outputs."
            raise ValueError(msg)
        if self.method is ResampleMethod.HERMITE and set(self.input_derivatives) != {1}:
            msg = "Hermite resampling requires exactly first-order derivative inputs."
            raise ValueError(msg)
        if self.method not in {ResampleMethod.LINEAR, ResampleMethod.HERMITE} and has_in:
            msg = f"{self.method.value} resampling does not accept input_derivatives."
            raise ValueError(msg)

        generated = set(self.position_columns)
        for order, cols in self.output_derivatives.items():
            invalid_overlap = (generated & set(cols)) - set(self.input_derivatives.get(order, []))
            if invalid_overlap:
                msg = "Output columns can only reuse names from matching input_derivatives."
                raise ValueError(msg)
            generated.update(cols)

        return self

    @override
    def __repr_args__(self) -> list[tuple[str, object]]:
        repr_args: list[tuple[str, object]] = []
        repr_args.extend([
            ("factors", f"{self.up}:{self.down}"),
            ("method", self.method.value),
            ("position_columns", tuple(self.position_columns.keys())),
            ("max_gap", self.max_gap),
            ("sort", self.sort),
        ])
        if self.input_derivatives:
            repr_args.append((
                "input_derivatives",
                {order: tuple(cols.keys()) for order, cols in self.input_derivatives.items()},
            ))
        if self.output_derivatives:
            repr_args.append((
                "output_derivatives",
                {order: tuple(cols.keys()) for order, cols in self.output_derivatives.items()},
            ))
        return repr_args

    @property
    def no_resampling(self) -> bool:
        """Whether the spec keeps the original sampling rate."""
        return self.up == 1 and self.down == 1


@dataclass(frozen=True)
class ResamplePlan:
    frame_column: str
    group_by: tuple[str, ...]
    segment_keys: tuple[str, ...]
    position_columns: ColumnOrder
    input_derivatives: DerivativeOrderMap
    output_derivatives: DerivativeOrderMap
    packed_columns: ColumnOrder
    emitted_columns: ColumnOrder
    evaluation_orders: tuple[int, ...]


def build_plan(
    spec: ResampleSpec, *, frame_column: str, group_by: str | Sequence[str] | None
) -> ResamplePlan:
    group_columns = normalize_group_by(group_by)
    packed_columns = dict.fromkeys((
        frame_column,
        *spec.position_columns,
        *chain.from_iterable(spec.input_derivatives.values()),
    ))
    emitted_columns = dict.fromkeys((
        *packed_columns.keys(),
        *chain.from_iterable(spec.output_derivatives.values()),
    ))

    evaluation_orders = tuple(
        dict.fromkeys((0, *spec.input_derivatives.keys(), *spec.output_derivatives.keys()))
    )

    return ResamplePlan(
        frame_column=frame_column,
        group_by=group_columns,
        segment_keys=(*group_columns, SEGMENT_COLUMN),
        position_columns=spec.position_columns,
        input_derivatives=dict(spec.input_derivatives),
        output_derivatives=dict(spec.output_derivatives),
        packed_columns=packed_columns,
        emitted_columns=emitted_columns,
        evaluation_orders=evaluation_orders,
    )


def segment_data(
    data: DataFrameT, *, frame_column: str, group_by: Sequence[str], max_gap: int, sort: bool
) -> DataFrameT:
    if sort:
        data = data.sort([*group_by, frame_column])
    expr = (pl.col(frame_column).diff() > max_gap).fill_null(value=False).cum_sum()
    if group_by:
        expr = expr.over(group_by)

    return data.with_columns(expr.alias(SEGMENT_COLUMN))


def packed_struct_expression(plan: ResamplePlan) -> pl.Expr:
    return pl.struct(pl.col(column) for column in plan.packed_columns)


def not_packed_columns(plan: ResamplePlan) -> pl.Expr:
    excluded = [*plan.group_by, *plan.emitted_columns, SEGMENT_COLUMN]
    return pl.all().exclude(excluded)
