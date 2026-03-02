from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import polars as pl
import polars.selectors as sl

from dronalize.common.trajectory.basic import collect

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.common.trajectory import T_DataFrame


@overload
def sliding_window(
    data: T_DataFrame,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    group_by: str | None = None,
    is_sorted: bool = False,
    include_boundaries: bool = False,
    return_iterable: Literal[True] = True,
) -> Iterable[pl.DataFrame]: ...


@overload
def sliding_window(
    data: T_DataFrame,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    group_by: str | None = None,
    is_sorted: bool = False,
    include_boundaries: bool = False,
    return_iterable: Literal[False],
) -> T_DataFrame: ...


def sliding_window(
    data: T_DataFrame,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    group_by: str | None = None,
    is_sorted: bool = False,
    include_boundaries: bool = False,
    return_iterable: bool = True,
) -> Iterable[pl.DataFrame] | T_DataFrame:
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
    return_iterable : bool, optional
        Whether to return an iterable of DataFrames or a single DataFrame
        containing all windows. Defaults to True.

    Returns
    -------
    Iterable[pl.DataFrame] or T_DataFrame
        DataFrames corresponding to each sliding window. Either as an
        iterable of DataFrames or a single DataFrame with all windows,
        depending on the `return_iterable` flag.

    """
    if not is_sorted:
        data = data.sort([sliding_col, group_by] if group_by else sliding_col)

    if return_iterable:
        return _sliding_window_iterable(
            collect(data),
            window_size,
            step_size,
            sliding_col,
            include_boundaries=include_boundaries,
        )

    return _sliding_window(
        data,
        window_size,
        step_size,
        sliding_col,
        include_boundaries=include_boundaries,
    )


def _sliding_window_iterable(
    data: pl.DataFrame,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    group_by: str | None = None,
    include_boundaries: bool = False,
) -> Iterable[pl.DataFrame]:
    for _, window in data.group_by_dynamic(
        sliding_col,
        every=f"{step_size}i",
        period=f"{window_size}i",
        include_boundaries=include_boundaries,
        group_by=group_by,
    ):
        if not window.is_empty():
            yield window


def _sliding_window(
    data: T_DataFrame,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    group_by: str | None = None,
    include_boundaries: bool = False,
) -> T_DataFrame:
    return (
        data
        .group_by_dynamic(
            sliding_col,
            every=f"{step_size}i",
            period=f"{window_size}i",
            include_boundaries=include_boundaries,
            group_by=group_by,
        )
        .agg(
            pl.col(sliding_col).alias(f"{sliding_col}_actual"),
            pl.all().exclude(sliding_col),
        )
        .with_row_index("window_index")
        .explode(sl.all().exclude("window_index", sliding_col))
        .drop(sliding_col)
        .rename({f"{sliding_col}_actual": sliding_col})
    )
