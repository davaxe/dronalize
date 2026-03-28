from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    """Estimate yaw from velocity vectors."""
    return data.with_columns(yaw_from_vel_expr(vx_col, vy_col, yaw_col))


def yaw_from_vel_expr(
    vx_col: str = "vx",
    vy_col: str = "vy",
    yaw_col: str = "yaw",
) -> pl.Expr:
    """Return a Polars expression estimating yaw from velocity vectors."""
    return pl.arctan2(pl.col(vy_col), pl.col(vx_col)).alias(yaw_col)


def yaw_from_pos(
    data: DataFrameT,
    x_col: str = "x",
    y_col: str = "y",
    yaw_col: str = "yaw",
) -> DataFrameT:
    """Estimate yaw from position differences."""
    return data.with_columns(yaw_from_pos_expr(x_col, y_col, yaw_col))


def yaw_from_pos_expr(
    x_col: str = "x",
    y_col: str = "y",
    yaw_col: str = "yaw",
) -> pl.Expr:
    """Return a Polars expression estimating yaw from position samples."""

    def get_diff_expr(col_name: str) -> pl.Expr:
        forward = pl.col(col_name).shift(-1) - pl.col(col_name)
        backward = pl.col(col_name) - pl.col(col_name).shift(1)
        central = (pl.col(col_name).shift(-1) - pl.col(col_name).shift(1)) / 2
        return central.fill_null(forward).fill_null(backward)

    x_diff = get_diff_expr(x_col)
    y_diff = get_diff_expr(y_col)
    return pl.arctan2(y_diff, x_diff).alias(yaw_col)


def derivative(
    data: DataFrameT,
    *x: str,
    dt: float = 1.0,
    n: int = 1,
    include_intermediate: bool = False,
    group_by: str | Sequence[str] | None = None,
    derivative_rename: dict[int, list[str]] | None = None,
) -> DataFrameT:
    """Compute finite-difference derivatives for one or more columns."""
    derivative_rename = {} if derivative_rename is None else derivative_rename

    def get_gradient_expr(input_expr: pl.Expr) -> pl.Expr:
        central = (input_expr.shift(-1) - input_expr.shift(1)) / (2 * dt)
        forward = (input_expr.shift(-1) - input_expr) / dt
        backward = (input_expr - input_expr.shift(1)) / dt
        expr = central.fill_null(forward).fill_null(backward)
        return expr.over(group_by) if group_by is not None else expr

    all_expressions: list[pl.Expr] = []
    current_exprs = [pl.col(col_name) for col_name in x]

    for i in range(1, n + 1):
        rename_list = derivative_rename.get(i, [f"d{i}_{original_root}" for original_root in x])

        next_order_exprs: list[pl.Expr] = []
        for j, expr in enumerate(current_exprs):
            grad_expr = get_gradient_expr(expr)
            aliased_expr = grad_expr.alias(rename_list[j])
            next_order_exprs.append(grad_expr)
            if i == n or include_intermediate:
                all_expressions.append(aliased_expr)

        current_exprs = next_order_exprs

    return data.with_columns(all_expressions)
