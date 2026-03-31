from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator, PPoly

from dronalize.processing.pipeline.functional.resample._common import (
    SEGMENT_COLUMN,
    ResamplePlan,
    ResampleSpec,
    build_plan,
    not_packed_columns,
    packed_struct_expression,
    segment_data,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize._internal.typing import DataFrameT

Interpolator = PPoly
InterpolatorFactory = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64] | None], Interpolator
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

    return (
        segmented
        .group_by(plan.segment_keys, maintain_order=True)
        .agg(
            packed_struct_expression(plan)
            .map_batches(
                lambda series: _resample_segment(
                    series,
                    plan=plan,
                    up=spec.up,
                    down=spec.down,
                    sample_time=spec.sample_time,
                    interpolator_factory=interpolator_factory,
                ),
                return_dtype=pl.Struct(_output_dtype(plan)),
            )
            .alias("_resampled"),
            not_packed_columns(plan).first(),
        )
        .explode("_resampled")
        .unnest("_resampled")
        .drop(SEGMENT_COLUMN, strict=False)
    )


def _output_dtype(plan: ResamplePlan) -> dict[str, type[pl.DataType]]:
    dtype: dict[str, type[pl.DataType]] = dict.fromkeys(plan.packed_columns, pl.Float64)
    dtype[plan.frame_column] = pl.Int64
    for columns in plan.output_derivatives.values():
        dtype.update(dict.fromkeys(columns, pl.Float64))

    return dtype


def cubic_spline_interpolator_factory(
    time_old: npt.NDArray[np.float64],
    position_old: npt.NDArray[np.float64],
    derivative_old: npt.NDArray[np.float64] | None = None,
) -> Interpolator:
    _ = derivative_old
    return CubicSpline(time_old, position_old, axis=0, bc_type="natural")


def pchip_interpolator_factory(
    time_old: npt.NDArray[np.float64],
    position_old: npt.NDArray[np.float64],
    derivative_old: npt.NDArray[np.float64] | None = None,
) -> Interpolator:
    _ = derivative_old
    return PchipInterpolator(time_old, position_old, axis=0)


def cubic_hermite_interpolator_factory(
    time_old: npt.NDArray[np.float64],
    position_old: npt.NDArray[np.float64],
    derivative_old: npt.NDArray[np.float64] | None = None,
) -> Interpolator:
    if derivative_old is None:
        msg = "Hermite resampling requires first-order derivative inputs."
        raise ValueError(msg)
    return CubicHermiteSpline(time_old, position_old, dydx=derivative_old, axis=0)


def _resample_segment(
    series: pl.Series,
    *,
    plan: ResamplePlan,
    up: int,
    down: int,
    sample_time: float,
    interpolator_factory: InterpolatorFactory,
) -> pl.Series:
    df = series.struct.unnest()

    if df.height <= 1:
        return _resample_single_point(series.name, df, plan, up=up, down=down)

    return _resample_multi_point(
        series.name,
        df,
        plan,
        up=up,
        down=down,
        sample_time=sample_time,
        interpolator_factory=interpolator_factory,
    )


def _resample_single_point(
    name: str, df: pl.DataFrame, plan: ResamplePlan, *, up: int, down: int
) -> pl.Series:
    out: dict[str, object] = {}
    output_order = tuple(plan.emitted_columns.keys())

    frames_old = df[plan.frame_column].to_numpy().astype(np.float64, copy=False)
    out[plan.frame_column] = (frames_old * (up / down)).astype(np.int64)

    positions = df.select(tuple(plan.position_columns)).to_numpy().astype(np.float64, copy=False)
    for i, column in enumerate(plan.position_columns):
        out[column] = positions[:, i]

    for order, columns in plan.output_derivatives.items():
        if plan.input_derivatives.get(order) == columns:
            values = df.select(tuple(columns)).to_numpy().astype(np.float64, copy=False)
        else:
            values = np.full((df.height, len(columns)), np.nan, dtype=np.float64)

        for i, column in enumerate(columns):
            out[column] = values[:, i]

    return pl.DataFrame(out).select(output_order).to_struct(name)


def _resample_multi_point(
    name: str,
    df: pl.DataFrame,
    plan: ResamplePlan,
    *,
    up: int,
    down: int,
    sample_time: float,
    interpolator_factory: InterpolatorFactory,
) -> pl.Series:
    frame_old = df[plan.frame_column].to_numpy().astype(np.float64, copy=False)
    time_old = frame_old * sample_time

    step_frames = down / up
    step_time = step_frames * sample_time
    time_start = float(time_old[0])
    time_end = float(time_old[-1])
    n_new = int(np.floor((time_end - time_start) / step_time)) + 1
    time_new = time_start + np.arange(n_new, dtype=np.float64) * step_time
    position_old = df.select(tuple(plan.position_columns)).to_numpy().astype(np.float64, copy=False)

    first_derivative_columns = plan.input_derivatives.get(1)
    first_derivative_old = (
        df.select(tuple(first_derivative_columns)).to_numpy().astype(np.float64, copy=False)
        if first_derivative_columns
        else None
    )

    interpolator = interpolator_factory(time_old, position_old, first_derivative_old)
    evaluated = {order: interpolator(time_new, nu=order) for order in plan.evaluation_orders}

    out: dict[str, object] = {}
    output_order = tuple(plan.emitted_columns.keys())
    out[plan.frame_column] = np.arange(n_new, dtype=np.int64) + int(frame_old[0] * up / down)
    for i, column in enumerate(plan.position_columns):
        out[column] = evaluated[0][:, i]

    for order, columns in plan.output_derivatives.items():
        values = evaluated[order]
        for i, column in enumerate(columns):
            out[column] = values[:, i]

    return pl.DataFrame(out).select(output_order).to_struct(name)
