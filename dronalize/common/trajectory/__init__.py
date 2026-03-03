from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import polars as pl

    DataFrameT = TypeVar("DataFrameT", pl.DataFrame, pl.LazyFrame)
