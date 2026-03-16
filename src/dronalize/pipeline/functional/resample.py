from __future__ import annotations

from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING, ClassVar, overload

import polars as pl
import polars.selectors as cs
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize._internal._types import DataFrameT


class ResamplingMethod(str, Enum):
    """Enumeration of resampling methods for trajectory data."""

    FAST = "fast"
    """Linear interpolation-based resampling."""


class Resampling(BaseModel):
    """Configuration for resampling trajectories."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    up: int = Field(gt=0, description="Upsampling factor.", default=1)
    down: int = Field(gt=0, description="Downsampling factor.", default=1)
    method: ResamplingMethod = Field(
        default=ResamplingMethod.FAST, description="Method used for resampling."
    )

    @model_validator(mode="after")
    def _simplify_ratio(self) -> Resampling:
        simplified_up, simplified_down = Fraction(self.up, self.down).as_integer_ratio()
        # setattr is needed since the model is frozen.
        object.__setattr__(self, "up", simplified_up)
        object.__setattr__(self, "down", simplified_down)
        return self

    @property
    def factors(self) -> tuple[int, int]:
        """(up, down) resampling factors."""
        return (self.up, self.down)

    @property
    def no_resampling(self) -> bool:
        """Whether no resampling is applied."""
        return self.up == 1 and self.down == 1


def resample(
    data: DataFrameT,
    resampling: Resampling,
    *,
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    velocity_columns: Sequence[str] = (),
    acceleration_columns: Sequence[str] = (),
    group_by: str | Sequence[str] | None = None,
) -> DataFrameT:
    """Resample trajectory data to a new sampling rate.

    This function adjusts the temporal resolution of track data by upsampling and
    downsampling. In the currently supported linear mode, position columns are
    linearly interpolated while velocity, acceleration, and other non-position
    columns are treated as piecewise-constant over each interval.

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
    velocity_columns : Sequence[str], optional
        Velocity columns to preserve using zero-order hold during upsampling.
        Defaults to an empty sequence.
    acceleration_columns : Sequence[str], optional
        Acceleration columns to preserve using zero-order hold during
        upsampling. Defaults to an empty sequence.
    group_by : str or Sequence[str], optional
        Column names used to partition data into individual tracks (e.g.,
        "track_id"). If None, data is treated as a single track.

    Returns
    -------
    T_DataFrame
        A Polars DataFrame containing the resampled data.

    """
    _ = velocity_columns, acceleration_columns
    method = resampling.method
    up, down = resampling.factors
    group_by = [group_by] if isinstance(group_by, str) else group_by
    if method is ResamplingMethod.FAST:
        return _resample_dataframe(
            data=data,
            up=up,
            down=down,
            frame_column=frame_column,
            pos_columns=pos_columns,
            group_by=group_by,
        )

    msg = f"Resampling method {method} is not implemented."
    raise NotImplementedError(msg)


def _resample_dataframe(
    data: DataFrameT,
    up: int,
    down: int,
    *,
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    group_by: Sequence[str] | None = None,
) -> DataFrameT:
    up, down = Fraction(up, down).as_integer_ratio()
    data_ = (
        data
        if up <= 1
        else _upsample_dataframe(
            data,
            up,
            frame_column,
            pos_columns=pos_columns,
            group_by=group_by,
        )
    )
    return data_ if down <= 1 else _downsample_dataframe(data_, down, frame_column)


def _downsample_dataframe(
    data: DataFrameT,
    factor: int,
    frame_column: str = "frame",
) -> DataFrameT:
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
    pos_columns: Sequence[str] = ("x", "y"),
    group_by: Sequence[str] | None = None,
) -> pl.DataFrame: ...


@overload
def _upsample_dataframe(
    data: pl.LazyFrame,
    factor: int,
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    group_by: Sequence[str] | None = None,
) -> pl.LazyFrame: ...


def _upsample_dataframe(
    data: pl.DataFrame | pl.LazyFrame,
    factor: int,
    frame_column: str = "frame",
    pos_columns: Sequence[str] = ("x", "y"),
    group_by: Sequence[str] | None = None,
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
    pos_columns : Sequence[str], optional
        Columns to linearly interpolate between existing samples.

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
    lf = data.lazy() if is_eager else data

    data_scaled = lf.with_columns(pl.col(frame_column) * factor)

    # Conditionally handle group_by to prevent ComputeError on empty groups
    if group_by:
        upsampled = (
            data_scaled
            .group_by(group_by)
            .agg(
                pl.int_range(
                    pl.col(frame_column).min(),
                    pl.col(frame_column).max() + 1,
                    step=1,
                    dtype=pl.Int32,
                ).alias(frame_column),
            )
            .explode(frame_column)
        )
    else:
        upsampled = data_scaled.select(
            pl.int_range(
                pl.col(frame_column).min(),
                pl.col(frame_column).max() + 1,
                step=1,
                dtype=pl.Int32,
            ).alias(frame_column)
        )

    on: list[str] = [*group_by, frame_column] if group_by else [frame_column]
    valid_pos_cols = [c for c in pos_columns if c not in on]

    exprs: list[pl.Expr] = []
    if valid_pos_cols:
        interp_expr = cs.by_name(*valid_pos_cols, require_all=False).interpolate()
        if group_by:
            interp_expr = interp_expr.over(group_by)
        exprs.append(interp_expr)

    exclude_cols = [*on, *valid_pos_cols]
    hold_expr = pl.all().exclude(*exclude_cols).forward_fill()
    if group_by:
        hold_expr = hold_expr.over(group_by)
    exprs.append(hold_expr)

    result = upsampled.join(data_scaled, on=on, how="left").sort(on).with_columns(exprs)

    return result.collect() if is_eager else result
