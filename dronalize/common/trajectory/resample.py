from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, cast, overload

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from dronalize.common.trajectory.derivative import derivative

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.common.trajectory import T_DataFrame


class ResamplingMethod(StrEnum):
    """Enumeration of resampling methods for trajectory data."""

    FAST = "fast"

    SPLINE = "spline"


@dataclass(slots=True, frozen=True)
class Resampling:
    """Configuration for resampling trajectories."""

    up: int
    """Upsampling factor."""
    down: int
    """Downsampling factor."""
    method: ResamplingMethod = ResamplingMethod.FAST
    """Method used for resampling."""

    def __init__(
        self,
        up: int,
        down: int,
        method: Literal["fast", "spline"] | ResamplingMethod = ResamplingMethod.FAST,
    ) -> None:
        """Simplify the resampling ratio to its smallest integer ratio form."""
        simplified_up, simplified_down = Fraction(up, down).as_integer_ratio()
        if method == "fast":
            method = ResamplingMethod.FAST
        elif method == "spline":
            method = ResamplingMethod.SPLINE
        object.__setattr__(self, "up", simplified_up)
        object.__setattr__(self, "down", simplified_down)
        object.__setattr__(self, "method", method)

    @property
    def factors(self) -> tuple[int, int]:
        """(up, down) resampling factors."""
        return (self.up, self.down)

    @property
    def no_resampling(self) -> bool:
        """Whether no resampling is applied."""
        return self.up == 1 and self.down == 1


def resample_tracks(
    data: T_DataFrame,
    resampling: Resampling,
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    group_by: str | Sequence[str] | None = None,
    *,
    add_derivative: bool = False,
    add_second_derivative: bool = False,
    dt: float = 1.0,
    derivative_rename: dict[int, list[str]] | None = None,
    forward_fill: Sequence[str] | None = None,
) -> T_DataFrame:
    """Resample trajectory data to a new sampling rate.

    This function adjusts the temporal resolution of track data by upsampling and
    downsampling. It supports basic linear-style resampling ("fast") or smooth
    interpolation ("spline"), with the ability to append velocity and acceleration
    vectors directly to the resulting DataFrame.

    Parameters
    ----------
    data : T_DataFrame
        Input trajectory data in a Polars-compatible DataFrame format.
    resampling : Resampling
        Configuration specifying upsampling/downsampling factors and method.
    frame_column : str, optional
        Name of the column representing time or frame index. Defaults to "frame".
    pos_columns : Sequence[str], optional
        Column names representing spatial coordinates (e.g., x, y, z).
        Defaults to ("x", "y").
    group_by : str or Sequence[str], optional
        Column names used to partition data into individual tracks (e.g.,
        "track_id"). If None, data is treated as a single track.
    add_derivative : bool, optional
        Whether to calculate and append the first derivative (velocity) of
        the position columns. Defaults to False.
    add_second_derivative : bool, optional
        Whether to calculate and append the second derivative (acceleration).
        Defaults to False.
    dt : float, optional
        Time step between original frames, used to scale derivatives
        appropriately. Defaults to 1.0.
    derivative_rename : dict[int, list[str]], optional
        Mapping to define custom names for the resulting derivative columns.
        Defaults to None.
    forward_fill : Sequence[str], optional
        Columns to forward fill instead of interpolate.

    Returns
    -------
    T_DataFrame
        A Polars DataFrame containing the resampled coordinates and any
        requested derivatives.

    """
    method = resampling.method
    up, down = resampling.factors
    group_by = [group_by] if isinstance(group_by, str) else group_by
    if method in {"fast", ResamplingMethod.FAST}:
        resampled = _resample_dataframe(
            data=data,
            up=up,
            down=down,
            frame_column=frame_column,
            group_by=group_by,
            forward_fill=forward_fill,
        )
        if add_second_derivative:
            resampled = derivative(
                resampled,
                *pos_columns,
                n=2,
                dt=dt,
                group_by=group_by,
                include_intermediate=add_derivative,
                derivative_rename=derivative_rename,
            )
        elif add_derivative:
            resampled = derivative(
                resampled,
                *pos_columns,
                n=1,
                dt=dt,
                group_by=group_by,
                include_intermediate=add_derivative,
                derivative_rename=derivative_rename,
            )
        return resampled

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

    Parameters
    ----------
    data : T_DataFrame
        Trajectory data.
    up : int
        Upsampling factor.
    down : int, optional
        Downsampling factor. Defaults to 1.
    frame_column : str, optional
        Column for frame index. Defaults to "frame".
    pos_columns : Sequence[str], optional
        Columns for position. Defaults to ("x", "y").
    group_by : Sequence[str], optional
        Columns to group by. Defaults to None.
    add_derivative : bool, optional
        Whether to add first derivative columns. Defaults to False.
    add_second_derivative : bool, optional
        Whether to add second derivative columns. Defaults to False.
    derivative_rename : dict[int, list[str]], optional
        Optional dictionary to rename derivative columns.

    Returns
    -------
    T_DataFrame
        Resampled trajectory data.

    """
    up, down = Fraction(up, down).as_integer_ratio()
    if down == 1 and up == 1 and not add_derivative and not add_second_derivative:
        return data

    group_cols = list(group_by or [])

    # Pack relevant columns for the spline calculation
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
    aggregations = [
        resampled_struct,
        _new_frames_expr(up, down, frame_column),
        pl.exclude(mod_cols).first(),
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
    out_data: dict[str, npt.NDArray[np.float32]] = {}
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
    fields = dict.fromkeys(pos_cols, pl.Float32)
    if add_derivative:
        fields.update(
            dict.fromkeys(derivative_rename.get(1, [f"v{c}" for c in pos_cols]), pl.Float32)
        )
    if add_second_derivative:
        fields.update(
            dict.fromkeys(derivative_rename.get(2, [f"a{c}" for c in pos_cols]), pl.Float32)
        )
    return pl.Struct(fields)


def _new_frames_expr(
    up: int,
    down: int,
    frame_column: str,
) -> pl.Expr:
    n_new_expr = ((pl.len() - 1) * up).floordiv(down) + 1
    start_expr = pl.when(pl.len() <= 1).then(pl.col(frame_column).first()).otherwise(pl.lit(0))
    end_expr = pl.when(pl.len() <= 1).then(start_expr + 1).otherwise(n_new_expr)
    return pl.int_range(start_expr, end_expr, dtype=pl.Int32).alias(frame_column)


def _apply_interpolation(
    pos: npt.NDArray[np.float32],
    vel: npt.NDArray[np.float32] | None,
    up: int,
    down: int,
    *,
    include_first_derivative: bool = False,
    include_second_derivative: bool = False,
) -> tuple[npt.NDArray[np.float32], ...]:

    n_old = len(pos)
    # Frequency ratio is up/down, so Period ratio (step size) is down/up
    step_size = down / up
    t = np.arange(n_old, dtype=np.float32)

    # Calculate exactly how many steps fit in the duration (n_old - 1)
    # Using floor ensures we don't extrapolate past the end.
    max_time = n_old - 1
    n_new = int(np.floor(max_time / step_size)) + 1
    # We generate indices 0, 1, ... M and scale them by step_size.
    t_new = np.arange(n_new, dtype=np.float32) * step_size

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
    data_ = data if up <= 1 else _upsample_dataframe(data, up, frame_column, group_by, forward_fill)
    return data_ if down <= 1 else _downsample_dataframe(data_, down, frame_column)


def _downsample_dataframe(
    data: T_DataFrame,
    factor: int,
    frame_column: str = "frame",
) -> T_DataFrame:
    """Downsample by an integer factor using decimation.

    Parameters
    ----------
    data : T_DataFrame
        DataFrame containing the tracks to be downsampled.
    factor : int
        Integer downsampling factor.
    frame_column : str, optional
        Column name for frame index. Defaults to "frame".

    Returns
    -------
    T_DataFrame
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


@overload
def _upsample_dataframe(
    data: pl.DataFrame,
    factor: int,
    frame_column: str = "frame",
    group_by: Sequence[str] | None = None,
    forward_fill: Sequence[str] | None = None,
) -> pl.DataFrame: ...


@overload
def _upsample_dataframe(
    data: pl.LazyFrame,
    factor: int,
    frame_column: str = "frame",
    group_by: Sequence[str] | None = None,
    forward_fill: Sequence[str] | None = None,
) -> pl.LazyFrame: ...


def _upsample_dataframe(
    data: T_DataFrame,
    factor: int,
    frame_column: str = "frame",
    group_by: Sequence[str] | None = None,
    forward_fill: Sequence[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Upsample by an integer factor using linear interpolation.

    Parameters
    ----------
    data : T_DataFrame
        DataFrame containing the tracks to be upsampled.
    factor : int
        Integer upsampling factor.
    frame_column : str, optional
        Column name for frame index. Defaults to "frame".
    group_by : Sequence[str], optional
        Columns to group by. Defaults to None.
    forward_fill : Sequence[str], optional
        Columns to forward fill instead of linear interpolation.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Upsampled DataFrame.

    """
    if factor == 1:
        return data
    if factor < 0:
        msg = "upsampling factor must be positive"
        raise ValueError(msg)
    is_eager = isinstance(data, pl.DataFrame)
    lf: pl.LazyFrame = cast("pl.LazyFrame", data.lazy() if is_eager else data)
    data_scaled = lf.with_columns(pl.col(frame_column) * factor)
    upsampled = (
        data_scaled
        .group_by(group_by)
        .agg(
            pl.int_range(
                pl.col(frame_column).min(),
                pl.col(frame_column).max() + 1,
                step=1,
                dtype=pl.Int32,
            ).alias(frame_column)
        )
        .explode(frame_column)
    )

    on: list[str] = [*group_by, frame_column] if group_by else [frame_column]
    exclude: list[str] = [*on, *(forward_fill or [])]
    interp_expr: pl.Expr = pl.all().exclude(exclude).interpolate()
    if group_by:
        interp_expr = interp_expr.over(group_by)

    exprs: list[pl.Expr] = [interp_expr]
    if forward_fill:
        forward_fill_exp = pl.col(forward_fill)
        if group_by:
            forward_fill_exp = forward_fill_exp.over(group_by)
        exprs.append(forward_fill_exp.forward_fill())
    result = upsampled.join(data_scaled, on=on, how="left").sort(on).with_columns(*exprs)

    if isinstance(result, pl.LazyFrame) and is_eager:
        return result.collect()
    return result


@overload
def cubic_spline_interpolation(
    t: npt.NDArray[np.float32],
    t_new: npt.NDArray[np.float32],
    pos: npt.NDArray[np.float32],
    *,
    vel: npt.NDArray[np.float32] | None = None,
    hermite: bool = False,
    include_first_derivative: Literal[False] = False,
    include_second_derivative: Literal[False] = False,
) -> npt.NDArray[np.float32]: ...


@overload
def cubic_spline_interpolation(
    t: npt.NDArray[np.float32],
    t_new: npt.NDArray[np.float32],
    pos: npt.NDArray[np.float32],
    *,
    vel: npt.NDArray[np.float32] | None = None,
    hermite: bool = False,
    include_first_derivative: Literal[True],
    include_second_derivative: Literal[False] = False,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...


@overload
def cubic_spline_interpolation(
    t: npt.NDArray[np.float32],
    t_new: npt.NDArray[np.float32],
    pos: npt.NDArray[np.float32],
    *,
    vel: npt.NDArray[np.float32] | None = None,
    hermite: bool = False,
    include_first_derivative: Literal[False] = False,
    include_second_derivative: Literal[True],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...


@overload
def cubic_spline_interpolation(
    t: npt.NDArray[np.float32],
    t_new: npt.NDArray[np.float32],
    pos: npt.NDArray[np.float32],
    *,
    vel: npt.NDArray[np.float32] | None = None,
    hermite: bool = False,
    include_first_derivative: Literal[True],
    include_second_derivative: Literal[True],
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]: ...


@overload
def cubic_spline_interpolation(
    t: npt.NDArray[np.float32],
    t_new: npt.NDArray[np.float32],
    pos: npt.NDArray[np.float32],
    *,
    vel: npt.NDArray[np.float32] | None = None,
    hermite: bool = False,
    include_first_derivative: bool = False,
    include_second_derivative: bool = False,
) -> npt.NDArray[np.float32] | tuple[npt.NDArray[np.float32], ...]: ...


def cubic_spline_interpolation(
    t: npt.NDArray[np.float32],
    t_new: npt.NDArray[np.float32],
    pos: npt.NDArray[np.float32],
    *,
    vel: npt.NDArray[np.float32] | None = None,
    hermite: bool = False,
    include_first_derivative: bool = False,
    include_second_derivative: bool = False,
) -> npt.NDArray[np.float32] | tuple[npt.NDArray[np.float32], ...]:
    """Interpolate positions using CubicSpline or CubicHermiteSpline.

    If `vel` is provided and `hermite=True`, Hermite spline interpolation is
    used. It is possible to return the first and second derivatives (velocity
    and acceleration) along with the interpolated positions by setting
    `include_first_derivative` and `include_second_derivative` to True.

    Parameters
    ----------
    t : ndarray of float32, shape (n_samples,)
        Original time points corresponding to `pos`.
    t_new : ndarray of float32, shape (n_new_samples,)
        New time points at which to interpolate.
    pos : ndarray of float32, shape (n_samples, n_dims)
        Original positions to interpolate.
    vel : ndarray of float32, shape (n_samples, n_dims), optional
        Original velocities corresponding to `pos`. Needed if `hermite=True`.
    hermite : bool, optional
        Whether to use Hermite spline interpolation.
    include_first_derivative : bool, optional
        Whether to include the first derivative in the output.
    include_second_derivative : bool, optional
        Whether to include the second derivative in the output.

    Returns
    -------
    ndarray or tuple of ndarray
        Interpolated positions, and optionally interpolated velocities and/or
        accelerations as a tuple.

    """
    if hermite and vel is not None:
        spline = CubicHermiteSpline(t, pos, vel)
    else:
        spline = CubicSpline(t, pos, bc_type="clamped")

    results = [spline(t_new).astype(np.float32)]
    for nu, include in [
        (1, include_first_derivative),
        (2, include_second_derivative),
    ]:
        if include:
            results.append(spline(t_new, nu).astype(np.float32))
    return tuple(results) if len(results) > 1 else results[0]
