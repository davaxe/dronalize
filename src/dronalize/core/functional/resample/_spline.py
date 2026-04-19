"""Spline-based interpolation backends for temporal trajectory resampling."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.interpolate import CubicSpline, PchipInterpolator, PPoly

from dronalize.core.functional.resample._common import (
    SEGMENT_COLUMN,
    ResamplePlan,
    ResampleSpec,
    not_packed_columns,
    packed_struct_expression,
    segment_data,
)

if TYPE_CHECKING:
    from dronalize.core.typing import DataFrameT

Interpolator = PPoly
InterpolatorFactory = Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], Interpolator]


def spline_resample(
    data: DataFrameT,
    spec: ResampleSpec,
    plan: ResamplePlan,
    *,
    interpolator_factory: InterpolatorFactory,
) -> DataFrameT:
    """Resample trajectory data with a SciPy spline interpolator."""
    if spec.no_resampling and not plan.emit_velocity and not plan.emit_acceleration:
        return data.sort([*plan.group_by, plan.frame_column]).drop(SEGMENT_COLUMN, strict=False)

    segmented = segment_data(
        data, frame_column=plan.frame_column, group_by=plan.group_by, max_gap=spec.max_gap
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
    dtype: dict[str, type[pl.DataType]] = {plan.frame_column: pl.Int64}
    for column in (*plan.coordinates, *plan.velocity_columns, *plan.acceleration_columns):
        dtype[column] = pl.Float64
    return dtype


def cubic_spline_interpolator_factory(
    time_old: npt.NDArray[np.float64], coordinates_old: npt.NDArray[np.float64]
) -> Interpolator:
    """Build a natural cubic-spline interpolator for one trajectory segment."""
    return CubicSpline(time_old, coordinates_old, axis=0, bc_type="natural")


def pchip_interpolator_factory(
    time_old: npt.NDArray[np.float64], coordinates_old: npt.NDArray[np.float64]
) -> Interpolator:
    """Build a monotone PCHIP interpolator for one trajectory segment."""
    return PchipInterpolator(time_old, coordinates_old, axis=0)


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
    frames_old = df[plan.frame_column].to_numpy().astype(np.float64, copy=False)
    coordinates_old = df.select(tuple(plan.coordinates)).to_numpy().astype(np.float64, copy=False)
    out: dict[str, object] = {plan.frame_column: (frames_old * (up / down)).astype(np.int64)}

    for index, column in enumerate(plan.coordinates):
        out[column] = coordinates_old[:, index]
    for columns in (plan.velocity_columns, plan.acceleration_columns):
        for column in columns:
            out[column] = np.full(df.height, np.nan, dtype=np.float64)

    return pl.DataFrame(out).select(plan.emitted_columns).to_struct(name)


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
    coordinates_old = df.select(tuple(plan.coordinates)).to_numpy().astype(np.float64, copy=False)

    step_frames = down / up
    step_time = step_frames * sample_time
    time_start = float(time_old[0])
    time_end = float(time_old[-1])
    n_new = int(np.floor((time_end - time_start) / step_time)) + 1
    time_new = time_start + np.arange(n_new, dtype=np.float64) * step_time
    interpolator = interpolator_factory(time_old, coordinates_old)

    out: dict[str, object] = {
        plan.frame_column: np.arange(n_new, dtype=np.int64) + int(frame_old[0] * up / down)
    }
    coordinates_new = interpolator(time_new)
    for index, column in enumerate(plan.coordinates):
        out[column] = coordinates_new[:, index]

    if plan.emit_velocity:
        velocity = interpolator(time_new, nu=1)
        for index, column in enumerate(plan.velocity_columns):
            out[column] = velocity[:, index]

    if plan.emit_acceleration:
        acceleration = interpolator(time_new, nu=2)
        for index, column in enumerate(plan.acceleration_columns):
            out[column] = acceleration[:, index]

    return pl.DataFrame(out).select(plan.emitted_columns).to_struct(name)
