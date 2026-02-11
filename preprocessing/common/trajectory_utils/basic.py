from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl

if TYPE_CHECKING:
    from preprocessing.common.trajectory_utils import T_DataFrame


def lazy(data: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Convert a DataFrame to a LazyFrame if necessary."""
    if isinstance(data, pl.DataFrame):
        return data.lazy()
    return data


def collect(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Resolve a LazyFrame to a DataFrame if necessary."""
    if isinstance(data, pl.LazyFrame):
        return cast("pl.DataFrame", data.collect())
    return data


def yaw_from_vel(
    data: T_DataFrame,
    vx_col: str = "vx",
    vy_col: str = "vy",
    yaw_col: str = "yaw",
) -> T_DataFrame:
    """Estimate yaw from velocity vector.

    Args:
        data: Input data frame or lazy frame.
        vx_col: Name of the column containing x velocity. Defaults to "vx".
        vy_col: Name of the column containing y velocity. Defaults to "vy".
        yaw_col: Name of the column to store the estimated yaw. Defaults to "yaw".

    Returns:
        Data frame or lazy frame with the estimated yaw column added.

    """
    return data.with_columns(yaw_from_vel_expr(vx_col, vy_col, yaw_col))


def yaw_from_vel_expr(
    vx_col: str = "vx",
    vy_col: str = "vy",
    yaw_col: str = "yaw",
) -> pl.Expr:
    """Expression to estimate yaw from velocity vector.

    Args:
        vx_col: Name of the column containing x velocity. Defaults to "vx".
        vy_col: Name of the column containing y velocity. Defaults to "vy".
        yaw_col: Name of the column to store the estimated yaw. Defaults to "yaw".

    Returns:
        Polars expression to compute the estimated yaw.

    """
    return pl.arctan2(pl.col(vy_col), pl.col(vx_col)).alias(yaw_col)


def yaw_from_pos(
    data: T_DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    yaw_col: str = "yaw",
) -> T_DataFrame:
    """Estimate yaw from position differences.

    Args:
        data: Input data frame or lazy frame.
        x_col: Name of the column containing x position. Defaults to "x".
        y_col: Name of the column containing y position. Defaults to "y".
        yaw_col: Name of the column to store the estimated yaw. Defaults to "yaw".

    Returns:
        Data frame or lazy frame with the estimated yaw column added.

    """
    return data.with_columns(yaw_from_pos_expr(x_col, y_col, yaw_col))


def yaw_from_pos_expr(
    x_col: str = "x",
    y_col: str = "y",
    yaw_col: str = "yaw",
) -> pl.Expr:
    """Expression to estimate yaw from position differences.

    Args:
        x_col: Name of the column containing x position. Defaults to "x".
        y_col: Name of the column containing y position. Defaults to "y".
        yaw_col: Name of the column to store the estimated yaw. Defaults to "yaw".

    Returns:
        Polars expression to compute the estimated yaw.

    """
    return pl.arctan2(
        pl.col(y_col).diff(),
        pl.col(x_col).diff(),
    ).alias(yaw_col)
