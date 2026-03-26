from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize._internal._typing import DataFrameT


def sliding_window(
    data: DataFrameT,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    group_by: str | Sequence[str] | None = None,
    is_sorted: bool = False,
    include_boundaries: bool = False,
    offset_sliding_col: bool = False,
) -> DataFrameT:
    """Generate sliding windows from a DataFrame.

    When returning as an iterable, the function yields DataFrames for each
    window. This means that if the input was a `pl.LazyFrame` it will be
    collected.

    Parameters
    ----------
    data : T_DataFrame
        Input DataFrame to generate windows from.
    window_size : int
        Number of rows in each window.
    step_size : int
        Number of rows to move the window at each step.
    sliding_col : str, optional
        Column name to use for determining the window boundaries.
        Defaults to "frame".
    group_by : str, optional
        Column name(s) to group by before applying the sliding window.
        This allows for generating windows within each group separately.
    is_sorted : bool, optional
        Whether the input DataFrame is already sorted by `sliding_col`.
        If False, the DataFrame will be sorted before generating windows.
        Defaults to False.
    include_boundaries : bool, optional
        Passed to `group_by_dynamic` to include window boundaries in the
        output. Defaults to False.

    Returns
    -------
    T_DataFrame
        Adds a `window_index` column to the input DataFrame indicating the window
        each row belongs to.
    """
    group_keys = [group_by] if isinstance(group_by, str) else list(group_by or [])

    if not is_sorted:
        data = data.sort([sliding_col, *group_keys] if group_keys else sliding_col)

    sliding_col_expr = pl.col(sliding_col)
    if offset_sliding_col:
        sliding_col_expr = sliding_col_expr.sub(sliding_col_expr.first())

    return (
        data
        .group_by_dynamic(
            sliding_col,
            every=f"{step_size}i",
            period=f"{window_size}i",
            include_boundaries=include_boundaries,
            group_by=group_keys or None,
        )
        .agg(
            sliding_col_expr.alias(f"{sliding_col}_actual"),
            pl.all().exclude(sliding_col),
        )
        .with_row_index("window_index")
        .explode(cs.all().exclude("window_index", sliding_col, *group_keys))
        .drop(sliding_col)
        .rename({f"{sliding_col}_actual": sliding_col})
    )
