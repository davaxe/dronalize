from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from preprocessing.trajectory.utils import derivative

if TYPE_CHECKING:
    from collections.abc import Sequence

T_DataFrame = TypeVar("T_DataFrame", pl.DataFrame, pl.LazyFrame)


def resample_tracks(
    data: T_DataFrame,
    up: int,
    down: int = 1,
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    group_by: str | Sequence[str] | None = None,
    *,
    add_derivative: bool = False,
    add_second_derivative: bool = False,
    dt: float = 1.0,
    method: Literal["fast", "spline"] = "fast",
    derivative_rename: dict[int, list[str]] | None = None,
) -> T_DataFrame:
    """Resample trajectory data to a new sampling rate.

    This function adjusts the temporal resolution of track data by upsampling and
    downsampling. It supports basic linear-style resampling ("fast") or smooth
    interpolation ("spline"), with the ability to append velocity and acceleration
    vectors directly to the resulting DataFrame.

    Args:
        data: Input trajectory data in a Polars-compatible DataFrame format.
        up: Upsampling factor (integer). Increases the number of samples.
        down: Downsampling factor (integer). Decreases the number of samples.
            Defaults to 1.
        frame_column: Name of the column representing time or frame index.
            Defaults to "frame".
        pos_columns: List of column names representing spatial coordinates
            (e.g., x, y, z). Defaults to ("x", "y").
        group_by: Column names used to partition data into individual tracks
            (e.g., "track_id"). If None, data is treated as a single track.
            Defaults to None.
        add_derivative: Whether to calculate and append the first derivative
            (velocity) of the position columns. Defaults to False.
        add_second_derivative: Whether to calculate and append the second
            derivative (acceleration). Defaults to False.
        dt: Time step between original frames, used to scale derivatives
            appropriately. Defaults to 1.0.
        method: Resampling algorithm to use. "fast" performs standard
            interpolation; "spline" uses cubic splines for smoother paths
            and more accurate derivatives. Defaults to "fast".
        derivative_rename: Mapping to define custom names for the resulting
            derivative columns. Defaults to None.

    Returns:
        A Polars DataFrame containing the resampled coordinates and any
        requested derivatives.

    """
    group_by = [group_by] if isinstance(group_by, str) else list(group_by or [])
    if method == "fast":
        # Just upsampling needed.
        upsampled = _resample_dataframe(
            data=data,
            up=up,
            down=down,
            frame_column=frame_column,
            group_by=group_by,
        )
        if add_second_derivative:
            upsampled = derivative(
                upsampled,
                *pos_columns,
                n=2,
                dt=dt,
                include_intermediate=add_derivative,
                derivative_rename=derivative_rename,
                group_by=group_by,
            )
        elif add_derivative:
            upsampled = derivative(
                upsampled,
                *pos_columns,
                n=1,
                dt=dt,
                include_intermediate=add_derivative,
                derivative_rename=derivative_rename,
                group_by=group_by,
            )
        return upsampled

    return _resample_dataframe_spline(
        data,
        up,
        down,
        frame_column=frame_column,
        pos_columns=pos_columns,
        group_by=group_by,
        add_derivative=add_derivative,
        add_second_derivative=add_second_derivative,
        derivative_rename=derivative_rename,
    )


def _resample_dataframe_spline(
    data: T_DataFrame,
    up: int,
    down: int = 1,
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    group_by: Sequence[str] | None = None,
    *,
    add_derivative: bool = False,
    add_second_derivative: bool = False,
    derivative_rename: dict[int, list[str]] | None = None,
) -> T_DataFrame:
    """Resample the track trajectories by an integer fraction.

    The required columns are:
    1. Column for frame index, specified by `frame_column`.
    2. Columns for positions to be interpolated, specified by `pos_columns`.

    To add estimated velocities calculated from the spline interpolation, set
    `add_velocity=True`. This will not do anything if velocity columns are
    already provided.

    Args:
        data: trajectory data.
        up: upsampling factor.
        down: downsampling factor. Defaults to 1.
        frame_column: column for frame index. Defaults to "frame".
        pos_columns: columns for position. Defaults to ("x", "y").
        group_by: columns to group by. Defaults to None.
        add_derivative: whether to add derivative columns if not provided.
            Defaults to False.
        add_second_derivative: whether to add second derivative columns if not provided.
            Defaults to False.
        derivative_rename: optional dictionary to rename derivative columns.

    Returns:
        Resampled trajectory data.

    """
    up, down = Fraction(up, down).as_integer_ratio()
    if down == 1 and up == 1 and not add_derivative and not add_second_derivative:
        return data

    group_cols = list(group_by or [])

    # Pack all relevant columns into a single struct for batch processing
    struct_input = pl.struct([pl.col(c) for c in pos_columns])
    resampled_struct = struct_input.map_batches(
        lambda series: _resample_batch(
            series,
            up,
            down,
            pos_cols=pos_columns,
            add_derivative=add_derivative,
            add_second_derivative=add_second_derivative,
            derivative_rename=derivative_rename,
        ),
        return_dtype=_return_type(
            add_derivative=add_derivative,
            add_second_derivative=add_second_derivative,
            pos_cols=pos_columns,
            derivative_rename=derivative_rename,
        ),
    ).alias("_resampled_batch")

    mod_cols = [*group_cols, *pos_columns, frame_column]
    other_cols = [c for c in data.columns if c not in mod_cols]
    aggregations = [
        resampled_struct,
        # Generate new frames to match the new length
        pl
        .col(frame_column)
        .map_batches(
            lambda series: _generate_new_frames(series, up, down),
            return_dtype=pl.Int32,
        )
        .alias(frame_column)
        .cast(pl.Int32),
        pl.col(other_cols).first(),
    ]

    return (
        data
        .group_by(group_cols)
        .agg(aggregations)
        .explode(["_resampled_batch", frame_column])
        .unnest("_resampled_batch")
    )


def _resample_batch(
    s: pl.Series,
    up: int,
    down: int,
    *,
    pos_cols: Sequence[str],
    add_derivative: bool = False,
    add_second_derivative: bool = False,
    derivative_rename: dict[int, list[str]] | None = None,
) -> pl.Series:
    if s.len() <= 1:
        return s

    derivative_rename = derivative_rename or {}

    df_struct = s.struct.unnest()
    pos_data = df_struct.select(pos_cols).to_numpy()
    vel_data = None

    res = _apply_interpolation(
        pos_data,
        vel_data,
        up,
        down,
        include_first_derivative=add_derivative,
        include_second_derivative=add_second_derivative,
    )

    pos_new = res[0]
    out_data: dict[str, npt.NDArray[np.float64]] = {}
    for i, col in enumerate(pos_cols):
        out_data[col] = pos_new[:, i]

    if len(res) > 1 and add_derivative:
        vel_new = res[1]
        target_cols = derivative_rename.get(1, [f"v{c}" for c in pos_cols])
        for i, col in enumerate(target_cols):
            out_data[col] = vel_new[:, i]

    if len(res) > 2 and add_second_derivative:
        acc_new = res[2]
        target_cols = derivative_rename.get(2, [f"a{c}" for c in pos_cols])
        for i, col in enumerate(target_cols):
            out_data[col] = acc_new[:, i]

    # Return as a Struct Series so it remains a single column in the aggregation
    return pl.DataFrame(out_data).to_struct(s.name)


def _return_type(
    *,
    add_derivative: bool,
    add_second_derivative: bool,
    pos_cols: Sequence[str],
    derivative_rename: dict[int, list[str]] | None = None,
) -> pl.Struct:
    derivative_rename = derivative_rename or {}
    fields = dict.fromkeys(pos_cols, pl.Float64)
    if add_derivative:
        fields.update(
            dict.fromkeys(
                derivative_rename.get(1, [f"v{c}" for c in pos_cols]), pl.Float64
            )
        )
    if add_second_derivative:
        fields.update(
            dict.fromkeys(
                derivative_rename.get(2, [f"a{c}" for c in pos_cols]), pl.Float64
            )
        )
    return pl.Struct(fields)


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
        return pl.Series(np.arange(n_new, dtype=np.int32))
    return pl.Series(np.arange(n_new, dtype=np.int32) * step_size)


def _apply_interpolation(
    pos: npt.NDArray[np.float64],
    vel: npt.NDArray[np.float64] | None,
    up: int,
    down: int,
    *,
    include_first_derivative: bool = False,
    include_second_derivative: bool = False,
) -> tuple[npt.NDArray[np.float64], ...]:

    n_old = len(pos)
    # Frequency ratio is up/down, so Period ratio (step size) is down/up
    step_size = down / up
    t = np.arange(n_old, dtype=np.float64)

    # Calculate exactly how many steps fit in the duration (n_old - 1)
    # Using floor ensures we don't extrapolate past the end.
    max_time = n_old - 1
    n_new = int(np.floor(max_time / step_size)) + 1
    # We generate indices 0, 1, ... M and scale them by step_size.
    t_new = np.arange(n_new, dtype=np.float64) * step_size

    res = cubic_spline_interpolation(
        t,
        t_new,
        pos,
        vel=vel,
        hermite=vel is not None,
        include_first_derivative=include_first_derivative,
        include_second_derivative=include_second_derivative,
    )
    return tuple(res) if isinstance(res, tuple) else (res,)


def _resample_dataframe(
    data: T_DataFrame,
    up: int,
    down: int,
    frame_column: str = "frame",
    group_by: Sequence[str] | None = None,
    forward_fill: Sequence[str] | None = None,
) -> T_DataFrame:
    up, down = Fraction(up, down).as_integer_ratio()
    data_ = (
        data
        if up <= 1
        else _upsample_dataframe(data, up, frame_column, group_by, forward_fill)
    )
    return data_ if down <= 1 else _downsample_dataframe(data_, down, frame_column)


def _downsample_dataframe(
    data: T_DataFrame,
    factor: int,
    frame_column: str = "frame",
) -> T_DataFrame:
    """Downsample by an integer factor using decimation.

    Args:
        data: DataFrame containing the tracks to be downsampled.
        factor: integer downsampling factor.
        frame_column: column name for frame index. Defaults to "frame".

    Returns:
        Downsampled DataFrame.

    """
    if factor == 1:
        return data
    if factor < 0:
        msg = "downsampling factor must be positive"
        raise ValueError(msg)

    return (
        data
        # Keep only frames that match the step factor
        .filter(pl.col(frame_column) % factor == 0)
        # Re-index the frames to be consecutive integers (0, 1, 2...)
        .with_columns((pl.col(frame_column) // factor).alias(frame_column))
    )


def _upsample_dataframe(
    data: T_DataFrame,
    factor: int,
    frame_column: str = "frame",
    group_by: Sequence[str] | None = None,
    forward_fill: Sequence[str] | None = None,
) -> T_DataFrame:
    """Upsample by an integer factor using linear interpolation.

    Args:
        data: DataFrame containing the tracks to be upsampled.
        factor: integer upsampling factor.
        frame_column: column name for frame index. Defaults to "frame".
        group_by: columns to group by. Defaults to None.
        forward_fill: columns to forward fill instead of linear interpolation.

    Returns:
        Upsampled DataFrame.

    """
    if factor == 1:
        return data
    if factor < 0:
        msg = "upsampling factor must be positive"
        raise ValueError(msg)

    is_eager = isinstance(data, pl.DataFrame)
    lf = data.lazy() if is_eager else data
    data_scaled = lf.with_columns(pl.col(frame_column) * factor)
    upsampled = (
        data_scaled
        .group_by(group_by)
        .agg(
            pl.int_range(
                pl.col(frame_column).min(),
                pl.col(frame_column).max() + 1,
                step=1,
            ).alias(frame_column)
        )
        .explode(frame_column)
    )

    on = [*group_by, frame_column] if group_by else [frame_column]
    exclude: list[str] = [*on, *(forward_fill or [])]
    result = (
        upsampled
        .join(data_scaled, on=on, how="left")
        .sort(on)
        .with_columns(
            pl.all().exclude(exclude).interpolate(),
            pl.col(forward_fill).forward_fill() if forward_fill else [],
        )
    )
    return result.collect() if is_eager else result  # pyright: ignore[reportReturnType]


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


@overload
def cubic_spline_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    vel: npt.NDArray[np.float64] | None = None,
    hermite: bool = False,
    include_first_derivative: bool = False,
    include_second_derivative: bool = False,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ...]: ...


def cubic_spline_interpolation(
    t: npt.NDArray[np.float64],
    t_new: npt.NDArray[np.float64],
    pos: npt.NDArray[np.float64],
    *,
    vel: npt.NDArray[np.float64] | None = None,
    hermite: bool = False,
    include_first_derivative: bool = False,
    include_second_derivative: bool = False,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ...]:
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
