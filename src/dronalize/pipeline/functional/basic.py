from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from dronalize._internal._typing import DataFrameT


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

    Uses `yaw_from_pos_expr`, which applies:
    - forward difference at the first row
    - central difference for interior rows
    - backward difference at the last row

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

    Examples
    --------
    Yaw should be 0 when moving in the positive x direction:
    >>> df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]})
    >>> yaw_from_pos(df)
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ x   ┆ y   ┆ yaw │
    │ --- ┆ --- ┆ --- │
    │ f64 ┆ f64 ┆ f64 │
    ╞═════╪═════╪═════╡
    │ 0.0 ┆ 0.0 ┆ 0.0 │
    │ 1.0 ┆ 0.0 ┆ 0.0 │
    │ 2.0 ┆ 0.0 ┆ 0.0 │
    └─────┴─────┴─────┘

    """
    return data.with_columns(yaw_from_pos_expr(x_col, y_col, yaw_col))


def yaw_from_pos_expr(
    x_col: str = "x",
    y_col: str = "y",
    yaw_col: str = "yaw",
) -> pl.Expr:
    """Estimate yaw from position samples for all rows.

    Uses:
    - forward difference at the first row
    - central difference for interior rows
    - backward difference at the last row

    Parameters
    ----------
    x_col : str, optional
        Name of the column containing x position. Defaults to "x".
    y_col : str, optional
        Name of the column containing y position. Defaults to "y".
    yaw_col : str, optional
        Name of the output yaw column. Defaults to "yaw".

    Returns
    -------
    pl.Expr
        Polars expression computing yaw for all rows.

    """

    def get_diff_expr(col_name: str) -> pl.Expr:
        forward = pl.col(col_name).shift(-1) - pl.col(col_name)
        backward = pl.col(col_name) - pl.col(col_name).shift(1)
        central = (pl.col(col_name).shift(-1) - pl.col(col_name).shift(1)) / 2
        return central.fill_null(forward).fill_null(backward)

    x_diff = get_diff_expr(x_col)
    y_diff = get_diff_expr(y_col)
    return pl.arctan2(y_diff, x_diff).alias(yaw_col)
