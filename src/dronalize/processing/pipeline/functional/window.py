from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
import polars.selectors as cs

from dronalize._internal.polars_ops import normalize_group_by

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize._internal.typing import DataFrameT

WindowPolicy = Literal["strict", "anchored", "partial"]


def sliding_window(
    data: DataFrameT,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    policy: WindowPolicy = "strict",
    group_by: str | Sequence[str] | None = None,
    is_sorted: bool = False,
    offset_sliding_col: bool = False,
) -> DataFrameT:
    """Generate sliding windows from a DataFrame.

    Parameters
    ----------
    data : DataFrameT
        Input DataFrame to generate windows from.
    window_size : int
        Temporal span of each window in units of `sliding_col`.
    step_size : int
        Distance between consecutive window starts in units of `sliding_col`.
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

    Returns
    -------
    DataFrameT
        Adds a `window_index` column to the input DataFrame indicating the window
        each row belongs to.
    """
    group_keys = list(normalize_group_by(group_by))

    if not is_sorted:
        data = data.sort([*group_keys, sliding_col] if group_keys else sliding_col)

    windows = _create_windows(
        data,
        window_size,
        step_size,
        sliding_col=sliding_col,
        policy=policy,
        group_by=group_keys,
        offset_sliding_col=offset_sliding_col,
    )
    return _explode_windows(windows, sliding_col, group_keys)


def _create_windows(
    data: DataFrameT,
    window_size: int,
    step_size: int,
    *,
    sliding_col: str,
    policy: WindowPolicy,
    group_by: list[str],
    offset_sliding_col: bool,
) -> DataFrameT:
    if policy == "anchored":
        max_expr = pl.col(sliding_col).max()
        if group_by:
            max_expr = max_expr.over(group_by)
        data = data.with_columns(max_expr.alias("_group_max"))

    slide_actual = pl.col(sliding_col)
    if offset_sliding_col:
        slide_actual -= slide_actual.first()

    grouped = data.group_by_dynamic(
        sliding_col, every=f"{step_size}i", period=f"{window_size}i", group_by=group_by or None
    ).agg(slide_actual.alias(f"{sliding_col}_actual"), cs.all().exclude(sliding_col))

    if policy == "strict":
        span = (
            pl.col(f"{sliding_col}_actual").list.last()
            - pl.col(f"{sliding_col}_actual").list.first()
            + 1
        )
        grouped = grouped.filter(span == window_size)
    elif policy == "anchored":
        grouped = grouped.filter(
            (pl.col(sliding_col) + window_size - 1) <= pl.col("_group_max").list.first()
        ).drop("_group_max")

    return grouped


def _explode_windows(
    windows: DataFrameT,
    sliding_col: str,
    group_keys: list[str],
    extra_exclude_cols: Sequence[str] = (),
) -> DataFrameT:
    """Explodes aggregated lists back to individual rows with a window index."""
    return (
        windows
        .with_row_index("window_index")
        .explode(cs.all().exclude("window_index", sliding_col, *group_keys, *extra_exclude_cols))
        .drop(sliding_col)
        .rename({f"{sliding_col}_actual": sliding_col})
    )
