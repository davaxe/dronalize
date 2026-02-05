from collections.abc import Sequence
from fractions import Fraction
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.interpolate import CubicHermiteSpline, CubicSpline


def resample_tracks(
    data: pl.DataFrame,
    ratio: float | Fraction,
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    vel_columns: Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
    *,
    add_velocity: bool = False,
) -> pl.DataFrame:
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
    down, up = fraction.numerator, fraction.denominator
    if down == 1 and up == 1:
        return data

    group_cols = [group_by] if isinstance(group_by, str) else list(group_by or [])
    aggregations: list[pl.Expr] = []
    res_col_names: list[str] = []

    for i, p_col in enumerate(pos_columns):
        v_col = vel_columns[i] if vel_columns and i < len(vel_columns) else None
        alias_name = f"_res_{p_col}"
        res_col_names.append(alias_name)

        struct_input = (
            pl.struct(pos=pl.col(p_col), vel=pl.col(v_col))
            if v_col
            else pl.struct(pos=pl.col(p_col))
        )

        aggregations.append(
            struct_input.map_batches(
                lambda series, v=v_col: _resample_batch(
                    series,
                    up,
                    down,
                    add_velocity=add_velocity,
                    has_velocity=v is not None,
                ),
            ).alias(alias_name)
        )

    # Extend the frame count to match the new number of samples after resampling
    aggregations.append(
        pl
        .col(frame_column)
        .map_batches(
            lambda series: pl.Series(
                np.arange(int(np.ceil(len(series) * up / down))) + series[0]
            )
        )
        .alias(frame_column)
    )

    res = (
        data
        .sort([*group_cols, frame_column])
        .group_by(group_cols)
        .agg(aggregations)
        .explode([*res_col_names, frame_column])
    )

    return _reform(res, pos_columns, vel_columns, add_velocity=add_velocity).drop(
        res_col_names
    )


def _reform(
    df: pl.DataFrame,
    pos_cols: Sequence[str],
    vel_cols: Sequence[str] | None = None,
    *,
    add_velocity: bool = False,
) -> pl.DataFrame:
    for i, p_col in enumerate(pos_cols):
        v_col = vel_cols[i] if vel_cols and i < len(vel_cols) else None
        struct_col = f"_res_{p_col}"
        df = df.with_columns(pl.col(struct_col).struct.field("pos").alias(p_col))
        if v_col:
            df = df.with_columns(pl.col(struct_col).struct.field("vel").alias(v_col))
        elif add_velocity:
            # If no v_col was provided but we calculated it, name it "v{p_col}"
            df = df.with_columns(
                pl.col(struct_col).struct.field("vel").alias(f"v{p_col}")
            )
    return df


def _apply_interpolation(
    pos: npt.NDArray[np.float64],
    vel: npt.NDArray[np.float64] | None,
    up: int,
    down: int,
    *,
    method: Literal["spline", "linear"] = "spline",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    n_old = len(pos)
    n_new = int(np.ceil(n_old * up / down))
    t = np.arange(n_old, dtype=np.float64)
    t_new = np.linspace(0, n_old - 1, n_new, dtype=np.float64)

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
    has_velocity: bool,
    add_velocity: bool = False,
) -> pl.Series:
    if s.len() <= 1:
        print(s)
        return s

    df_struct = s.struct.unnest()
    pos_res, vel_res = _apply_interpolation(
        df_struct["pos"].to_numpy(),
        df_struct["vel"].to_numpy() if has_velocity else None,
        up,
        down,
    )
    if has_velocity:
        return pl.Series([
            {"pos": p, "vel": v} for p, v in zip(pos_res, vel_res, strict=True)
        ])

    if add_velocity:
        return pl.Series([
            {"pos": p, "vel": v} for p, v in zip(pos_res, vel_res, strict=True)
        ])
    return pl.Series([{"pos": p} for p in pos_res])
