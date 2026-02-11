from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import polars as pl

    T_DataFrame = TypeVar("T_DataFrame", pl.DataFrame, pl.LazyFrame)
