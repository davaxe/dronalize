from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from dronalize.common.trajectory import DataFrameT


def lazy(data: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Convert a DataFrame to a LazyFrame if necessary."""
    if isinstance(data, pl.DataFrame):
        return data.lazy()
    return data


def collect(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Resolve a LazyFrame to a DataFrame if necessary."""
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data


def yaw_from_vel(
    data: DataFrameT,
    vx_col: str = "vx",
    vy_col: str = "vy",
    yaw_col: str = "yaw",
) -> DataFrameT:
    """Estimate yaw from velocity vector.

    Parameters
    ----------
    data : T_DataFrame
        Input data frame or lazy frame.
    vx_col : str, optional
        Name of the column containing x velocity. Defaults to "vx".
    vy_col : str, optional
        Name of the column containing y velocity. Defaults to "vy".
    yaw_col : str, optional
        Name of the column to store the estimated yaw. Defaults to "yaw".

    Returns
    -------
    T_DataFrame
        Data frame or lazy frame with the estimated yaw column added.

    """
    return data.with_columns(yaw_from_vel_expr(vx_col, vy_col, yaw_col))


def yaw_from_vel_expr(
    vx_col: str = "vx",
    vy_col: str = "vy",
    yaw_col: str = "yaw",
) -> pl.Expr:
    """Expression to estimate yaw from velocity vector.

    Parameters
    ----------
    vx_col : str, optional
        Name of the column containing x velocity. Defaults to "vx".
    vy_col : str, optional
        Name of the column containing y velocity. Defaults to "vy".
    yaw_col : str, optional
        Name of the column to store the estimated yaw. Defaults to "yaw".

    Returns
    -------
    pl.Expr
        Polars expression to compute the estimated yaw.

    """
    return pl.arctan2(pl.col(vy_col), pl.col(vx_col)).alias(yaw_col)


def yaw_from_pos(
    data: DataFrameT,
    x_col: str = "x",
    y_col: str = "y",
    yaw_col: str = "yaw",
) -> DataFrameT:
    """Estimate yaw from position differences.

    Parameters
    ----------
    data : T_DataFrame
        Input data frame or lazy frame.
    x_col : str, optional
        Name of the column containing x position. Defaults to "x".
    y_col : str, optional
        Name of the column containing y position. Defaults to "y".
    yaw_col : str, optional
        Name of the column to store the estimated yaw. Defaults to "yaw".

    Returns
    -------
    T_DataFrame
        Data frame or lazy frame with the estimated yaw column added.

    """
    return data.with_columns(yaw_from_pos_expr(x_col, y_col, yaw_col))


def yaw_from_pos_expr(
    x_col: str = "x",
    y_col: str = "y",
    yaw_col: str = "yaw",
) -> pl.Expr:
    """Expression to estimate yaw from position differences.

    Parameters
    ----------
    x_col : str, optional
        Name of the column containing x position. Defaults to "x".
    y_col : str, optional
        Name of the column containing y position. Defaults to "y".
    yaw_col : str, optional
        Name of the column to store the estimated yaw. Defaults to "yaw".

    Returns
    -------
    pl.Expr
        Polars expression to compute the estimated yaw.

    """
    return pl.arctan2(
        pl.col(y_col).diff(),
        pl.col(x_col).diff(),
    ).alias(yaw_col)
