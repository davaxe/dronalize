from collections.abc import Sequence
from fractions import Fraction
from typing import Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.interpolate import CubicHermiteSpline, CubicSpline

T_DataFame = TypeVar("T_DataFame", pl.DataFrame, pl.LazyFrame)


def resample_tracks(
    data: T_DataFame,
    ratio: float | Fraction,
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    vel_columns: Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
    *,
    add_velocity: bool = False,
) -> T_DataFame:
    """Resample the track trajectories by a fraction.

    The required columns are:
    1. Column for frame index, specified by `frame_column`.
    2. Columns for positions to be interpolated, specified by `pos_columns`.
    3. (Optional) Columns for velocities, specified by `vel_columns`. If
        provided, these are used for Hermite spline interpolation. If not provided,
        cubic spline interpolation is used instead.

    To add estimated velocities calculated from the spline interpolation, set
    `add_velocity=True`. This will not do anything if velocity columns are
    already provided.

    Args:
        data: trajectory data.
        ratio: ratio of new sampling frequency.
        frame_column: column for frame index. Defaults to "frame".
        pos_columns: columns for position. Defaults to ("x", "y").
        vel_columns: columns for velocity. If provided, they are used for Hermite
            spline interpolation.
        group_by: columns to group by. Defaults to None.
        add_velocity: whether to add velocity columns if not provided. Defaults to False.

    Returns:
        Resampled trajectory data.

    """
    if pos_columns is None and vel_columns is not None:
        msg = "Position columns must be provided if velocity columns are provided."
        raise ValueError(msg)

    fraction = Fraction(ratio).limit_denominator()
    up, down = fraction.numerator, fraction.denominator
    if down == 1 and up == 1 and not add_velocity:
        return data

    group_cols = [group_by] if isinstance(group_by, str) else list(group_by or [])

    # Pack all relevant columns into a single struct for batch processing
    input_cols = [pl.col(c) for c in pos_columns]
    if vel_columns:
        input_cols.extend([pl.col(c) for c in vel_columns])

    struct_input = pl.struct(input_cols)
    resampled_struct = struct_input.map_batches(
        lambda series: _resample_batch(
            series,
            up,
            down,
            pos_cols=pos_columns,
            vel_cols=vel_columns,
            add_velocity=add_velocity,
        )
    ).alias("_resampled_batch")

    aggregations = [
        resampled_struct,
        # Generate new frames to match the new length
        pl
        .col(frame_column)
        .map_batches(lambda series: _generate_new_frames(series, up, down))
        .alias(frame_column)
        .cast(pl.Int32),
    ]

    return (
        data
        .sort([*group_cols, frame_column])
        .group_by(group_cols)
        .agg(aggregations)
        .explode(["_resampled_batch", frame_column])
        .unnest("_resampled_batch")
    )


def _generate_new_frames(
    series: pl.Series,
    up: int,
    down: int,
    *,
    integer: bool = True,
) -> pl.Series:
    n_old = series.len()
    if n_old <= 1:
        return series

    max_time = n_old - 1
    step_size = down / up
    n_new = int(np.floor(max_time / step_size)) + 1
    if integer:
        return pl.Series(np.arange(n_new))
    return pl.Series(np.arange(n_new, dtype=np.int32) * step_size)


def _apply_interpolation(
    pos: npt.NDArray[np.float64],
    vel: npt.NDArray[np.float64] | None,
    up: int,
    down: int,
    *,
    method: Literal["spline", "linear"] = "spline",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    n_old = len(pos)
    # Frequency ratio is up/down, so Period ratio (step size) is down/up
    step_size = down / up
    t = np.arange(n_old, dtype=np.float64)

    # We calculate exactly how many steps fit in the duration (n_old - 1)
    # Using floor ensures we don't extrapolate past the end.
    max_time = n_old - 1
    n_new = int(np.floor(max_time / step_size)) + 1
    # We generate indices 0, 1, ... M and scale them by step_size.
    t_new = np.arange(n_new, dtype=np.float64) * step_size

    if method == "spline":
        pos_new, vel_new = cubic_spline_interpolation(
            t,
            t_new,
            pos,
            vel=vel,
            hermite=vel is not None,
            include_first_derivative=True,
        )
    elif method == "linear":
        pos_new, vel_new = linear_interpolation(
            t,
            t_new,
            pos,
            include_first_derivative=True,
        )
    return pos_new, vel_new


@overload
def cubic_spline_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    vel: npt.NDArray[np.float64] | None = None,
    hermite: bool = False,
    include_first_derivative: Literal[False] = False,
    include_second_derivative: Literal[False] = False,
) -> npt.NDArray[np.float64]: ...


@overload
def cubic_spline_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    vel: npt.NDArray[np.float64] | None = None,
    hermite: bool = False,
    include_first_derivative: Literal[True],
    include_second_derivative: Literal[False] = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


@overload
def cubic_spline_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    vel: npt.NDArray[np.float64],
    hermite: bool = False,
    include_first_derivative: Literal[False] = False,
    include_second_derivative: Literal[True],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


@overload
def cubic_spline_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    vel: npt.NDArray[np.float64] | None = None,
    hermite: bool = False,
    include_first_derivative: Literal[True],
    include_second_derivative: Literal[True],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]: ...


def cubic_spline_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    vel: npt.NDArray[np.float64] | None = None,
    hermite: bool = False,
    include_first_derivative: bool = False,
    include_second_derivative: bool = False,
) -> tuple[npt.NDArray[np.float64], ...] | npt.NDArray[np.float64]:
    """Interpolate positions using CubicSpline or CubicHermiteSpline.

    If `vel` is provided and `hermite=True`, Hermite spline interpolation is
    used. It is possible to return the first and second derivatives (velocity
    and acceleration) along with the interpolated positions by setting
    `include_first_derivative` and `include_second_derivative` to True.


    Args:
        pos: original positions to interpolate, shape (n_samples, n_dims).
        t: original time points corresponding to `pos`, shape (n_samples,).
        t_new: new time points at which to interpolate, shape (n_new_samples,).
        vel: original velocities corresponding to `pos`, shape (n_samples, n_dims).
            Needed if `hermite=True`.
        hermite: whether to use Hermite spline interpolation.
        include_first_derivative: whether to include the first derivative in the
            output.
        include_second_derivative: whether to include the second derivative in
            the output.

    Returns:
        (interpolated positions, [optional] interpolated velocities, [optional]
        interpolated accelerations)

    """
    if hermite and vel is not None:
        spline = CubicHermiteSpline(t, pos, vel)
    else:
        spline = CubicSpline(t, pos, bc_type="clamped")

    results = [spline(t_new)]
    for nu, include in [
        (1, include_first_derivative),
        (2, include_second_derivative),
    ]:
        if include:
            results.append(spline(t_new, nu))
    return tuple(results) if len(results) > 1 else results[0]


@overload
def linear_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    include_first_derivative: Literal[False] = False,
    include_second_derivative: Literal[False] = False,
) -> npt.NDArray[np.float64]: ...


@overload
def linear_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    include_first_derivative: Literal[True],
    include_second_derivative: Literal[False] = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


@overload
def linear_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    include_first_derivative: Literal[False] = False,
    include_second_derivative: Literal[True],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


@overload
def linear_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    include_first_derivative: Literal[True],
    include_second_derivative: Literal[True],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]: ...


def linear_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    include_first_derivative: bool = False,
    include_second_derivative: bool = False,
) -> tuple[npt.NDArray[np.float64], ...] | npt.NDArray[np.float64]:
    """Linearly interpolate positions.

    The velocity and acceleration are estimated using finite differences on the
    interpolated positions, so they may be noisy and not very accurate.

    Args:
        pos: original positions to interpolate, shape (n_samples, n_dims).
        t: original time points corresponding to `pos`, shape (n_samples,).
        t_new: new time points at which to interpolate, shape (n_new_samples,).
        include_first_derivative: whether to include the first derivative in the
            output. Defaults to False.
        include_second_derivative: whether to include the second derivative in the
            output. Defaults to False.

    Returns:
        (interpolated positions, [optional] interpolated velocities, [optional]
        interpolated accelerations)

    """
    # Estimate velocity and acceleration using finite differences
    pos = np.interp(t_new, t, pos)
    results = [pos]
    if include_first_derivative:
        vel = np.gradient(pos, t_new)
        results.append(vel)
    if include_second_derivative:
        acc = np.gradient(
            results[-1] if include_first_derivative else np.gradient(pos, t_new),
            t_new,
        )
        results.append(acc)
    return tuple(results) if len(results) > 1 else results[0]


def _resample_batch(
    s: pl.Series,
    up: int,
    down: int,
    *,
    pos_cols: Sequence[str],
    vel_cols: Sequence[str] | None,
    add_velocity: bool = False,
) -> pl.Series:
    if s.len() <= 1:
        return s

    df_struct = s.struct.unnest()
    pos_data = df_struct.select(pos_cols).to_numpy()
    vel_data = None
    if vel_cols:
        vel_data = df_struct.select(vel_cols).to_numpy()

    pos_new, vel_new = _apply_interpolation(
        pos_data,
        vel_data,
        up,
        down,
    )

    out_data: dict[str, npt.NDArray[np.float64]] = {}
    for i, col in enumerate(pos_cols):
        out_data[col] = pos_new[:, i]

    # Handle velocities
    if vel_cols:
        # If velocity columns existed, update them
        for i, col in enumerate(vel_cols):
            out_data[col] = vel_new[:, i]
    elif add_velocity:
        # If requested to add velocity, name them "v{pos_col}"
        for i, col in enumerate(pos_cols):
            col_name = vel_cols[i] if vel_cols and i < len(vel_cols) else f"v{col}"
            out_data[col_name] = vel_new[:, i]

    # Return as a Struct Series so it remains a single column in the aggregation
    return pl.DataFrame(out_data).to_struct(s.name)
