from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.common.trajectory import T_DataFrame


def derivative(
    data: T_DataFrame,
    *x: str,
    dt: float = 1.0,
    n: int = 1,
    include_intermediate: bool = False,
    group_by: str | Sequence[str] | None = None,
    derivative_rename: dict[int, list[str]] | None = None,
) -> T_DataFrame:
    """Compute the n-th order derivative for columns using finite differences.

    Calculates numerical derivatives for one or multiple columns. If multiple
    orders are requested via `include_intermediate`, all steps from $1$ to $n$
    are returned.

    .. note::
        Groups with only one datapoint will produce null values for the
        derivative columns, as finite differences require at least two points.

    Parameters
    ----------
    data : T_DataFrame
        Input DataFrame or LazyFrame.
    *x : str
        Names of columns to differentiate.
    dt : float, optional
        Constant time step between samples. Defaults to 1.0.
    n : int, optional
        The maximum order of the derivative. Defaults to 1.
    include_intermediate : bool, optional
        If True, retains all derivatives from 1 to n-1.
    group_by : str or Sequence[str], optional
        Column(s) used to partition data before calculation.
    derivative_rename : dict[int, list[str]], optional
        Mapping of order to a list of new column names.
        Format: `{order: [name_x1, name_x2, ...]}`.

    Returns
    -------
    T_DataFrame
        The original data structure with new derivative columns appended.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {
    ...         "time": [0, 1, 2, 3, 4],
    ...         "position": [0, 1, 4, 9, 16],
    ...     }
    ... )
    >>> derivative(df, "position", dt=1.0, n=2, include_intermediate=True)
    shape: (5, 4)
    ┌──────┬──────────┬─────────────┬─────────────┐
    │ time ┆ position ┆ d1_position ┆ d2_position │
    │ ---  ┆ ---      ┆ ---         ┆ ---         │
    │ i64  ┆ i64      ┆ f64         ┆ f64         │
    ╞══════╪══════════╪═════════════╪═════════════╡
    │ 0    ┆ 0        ┆ 1.0         ┆ 1.0         │
    │ 1    ┆ 1        ┆ 2.0         ┆ 1.5         │
    │ 2    ┆ 4        ┆ 4.0         ┆ 2.0         │
    │ 3    ┆ 9        ┆ 6.0         ┆ 1.5         │
    │ 4    ┆ 16       ┆ 7.0         ┆ 1.0         │
    └──────┴──────────┴─────────────┴─────────────┘

    """
    derivative_rename = {} if derivative_rename is None else derivative_rename

    def get_gradient_expr(input_expr: pl.Expr) -> pl.Expr:
        """Finite difference logic that nests expressions."""
        central = (input_expr.shift(-1) - input_expr.shift(1)) / (2 * dt)
        forward = (input_expr.shift(-1) - input_expr) / dt
        backward = (input_expr - input_expr.shift(1)) / dt
        expr = central.fill_null(forward).fill_null(backward)
        return expr.over(group_by) if group_by is not None else expr

    all_expressions: list[pl.Expr] = []

    # Track the actual Expr object, not the string name
    current_exprs = [pl.col(col_name) for col_name in x]

    for i in range(1, n + 1):
        rename_list = derivative_rename.get(i, [f"d{i}_{original_root}" for original_root in x])

        next_order_exprs = []
        for j, expr in enumerate(current_exprs):
            # Compute the next derivative using the current expression object
            grad_expr = get_gradient_expr(expr)
            aliased_expr = grad_expr.alias(rename_list[j])
            next_order_exprs.append(grad_expr)
            if i == n or include_intermediate:
                all_expressions.append(aliased_expr)

        current_exprs = next_order_exprs

    return data.with_columns(all_expressions)
