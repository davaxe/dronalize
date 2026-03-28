from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator, PPoly

from dronalize.processing.pipeline.functional.resample._common import (
    SEGMENT_COLUMN,
    ColumnOrder,
    DerivativeOrderMap,
    ResamplePlan,
    ResampleSpec,
    build_plan,
    not_packed_columns,
    packed_struct_expression,
    segment_data,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize._internal._typing import DataFrameT

Interpolator = PPoly
InterpolatorFactory = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64] | None],
    Interpolator,
]


def spline_resample(
    data: DataFrameT,
    spec: ResampleSpec,
    *,
    frame_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
    interpolator_factory: InterpolatorFactory,
) -> DataFrameT:
    plan = build_plan(spec, frame_column=frame_column, group_by=group_by)

    if spec.no_resampling and not plan.output_derivatives:
        base = data.sort([*plan.group_by, frame_column]) if spec.sort else data
        return base.drop(SEGMENT_COLUMN, strict=False)

    segmented = segment_data(
        data,
        frame_column=frame_column,
        group_by=plan.group_by,
        max_gap=spec.max_gap,
        sort=spec.sort,
    )
    packed_expr = packed_struct_expression(plan)

    return (
        segmented
        .group_by(plan.segment_keys, maintain_order=True)
        .agg(
            packed_expr.map_batches(
                lambda series: _resample_segment_struct(
                    series,
                    plan=plan,
                    up=spec.up,
                    down=spec.down,
                    interpolator_factory=interpolator_factory,
                ),
                return_dtype=pl.Struct(_return_dtype(plan)),
            ).alias("_resampled"),
            not_packed_columns(plan).first(),
        )
        .explode("_resampled")
        .unnest("_resampled")
        .drop(SEGMENT_COLUMN, strict=False)
    )


def _return_dtype(
    plan: ResamplePlan,
) -> dict[str, type[pl.DataType]]:
    base: dict[str, type[pl.DataType]] = {
        **dict.fromkeys(plan.packed_columns, pl.Float64),
    }
    base[plan.frame_column] = pl.Int64
    if plan.output_derivatives:
        for columns in plan.output_derivatives.values():
            base.update(dict.fromkeys(columns, pl.Float64))

    return base


def cubic_spline_interpolator_factory(
    t_old: npt.NDArray[np.float64],
    position_old: npt.NDArray[np.float64],
    derivative_old: npt.NDArray[np.float64] | None = None,
) -> Interpolator:
    _ = derivative_old
    return CubicSpline(t_old, position_old, axis=0, bc_type="natural")


def pchip_interpolator_factory(
    t_old: npt.NDArray[np.float64],
    position_old: npt.NDArray[np.float64],
    derivative_old: npt.NDArray[np.float64] | None = None,
) -> Interpolator:
    _ = derivative_old
    return PchipInterpolator(t_old, position_old, axis=0)


def cubic_hermite_interpolator_factory(
    t_old: npt.NDArray[np.float64],
    position_old: npt.NDArray[np.float64],
    derivative_old: npt.NDArray[np.float64] | None = None,
) -> Interpolator:
    if derivative_old is None:
        msg = "Hermite resampling requires first-order derivative inputs."
        raise ValueError(msg)
    return CubicHermiteSpline(t_old, position_old, dydx=derivative_old, axis=0)


def _resample_segment_struct(
    series: pl.Series,
    *,
    plan: ResamplePlan,
    up: int,
    down: int,
    interpolator_factory: InterpolatorFactory,
) -> pl.Series:
    df = series.struct.unnest()
    n_old = df.height
    out_data: dict[str, object] = {}
    frame_column = plan.frame_column
    position_columns = plan.position_columns
    input_derivatives = plan.input_derivatives
    output_derivatives = plan.output_derivatives
    evaluation_orders = plan.evaluation_orders
    output_order = df.columns

    if n_old <= 1:
        out_data[frame_column] = _single_point_frames(df, frame_column, up, down)
        _populate_single_point_outputs(
            out_data,
            df=df,
            position_columns=position_columns,
            input_derivatives=input_derivatives,
            output_derivatives=output_derivatives,
        )
        return pl.DataFrame(out_data).select(output_order).to_struct(series.name)

    t_old = df[frame_column].to_numpy().astype(np.float64, copy=False)
    step = down / up
    t_min = float(t_old[0])
    t_max = float(t_old[-1])
    n_new = int(np.floor((t_max - t_min) / step)) + 1
    t_new = t_min + np.arange(n_new, dtype=np.float64) * step

    positions_old = df.select(tuple(position_columns)).to_numpy().astype(np.float64, copy=False)
    first_derivative_columns = input_derivatives.get(1)
    first_derivative_old = (
        df.select(tuple(first_derivative_columns)).to_numpy().astype(np.float64, copy=False)
        if first_derivative_columns
        else None
    )
    interpolator = interpolator_factory(t_old, positions_old, first_derivative_old)
    evaluated = {order: interpolator(t_new, nu=order) for order in evaluation_orders}

    for index, column in enumerate(position_columns):
        out_data[column] = evaluated[0][:, index]

    for order, columns in output_derivatives.items():
        values = evaluated[order]
        for index, column in enumerate(columns):
            out_data[column] = values[:, index]

    out_data[frame_column] = np.arange(n_new, dtype=np.int64) + int(t_min * up / down)
    return pl.DataFrame(out_data).select(output_order).to_struct(series.name)


def _populate_single_point_outputs(
    out_data: dict[str, object],
    *,
    df: pl.DataFrame,
    position_columns: ColumnOrder,
    input_derivatives: DerivativeOrderMap,
    output_derivatives: DerivativeOrderMap,
) -> None:
    position_values = df.select(tuple(position_columns)).to_numpy().astype(np.float64, copy=False)
    for index, column in enumerate(position_columns):
        out_data[column] = position_values[:, index]

    for order, columns in output_derivatives.items():
        if input_derivatives.get(order) == columns:
            values = df.select(tuple(columns)).to_numpy().astype(np.float64, copy=False)
        else:
            values = np.full((df.height, len(columns)), np.nan, dtype=np.float64)
        for index, column in enumerate(columns):
            out_data[column] = values[:, index]


def _single_point_frames(
    df: pl.DataFrame,
    frame_column: str,
    up: int,
    down: int,
) -> npt.NDArray[np.int64]:
    frames = df[frame_column].to_numpy().astype(np.float64, copy=False)
    return (frames * (up / down)).astype(np.int64)
