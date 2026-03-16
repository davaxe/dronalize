from __future__ import annotations

from typing import ParamSpec, TypeVar

import polars as pl

DataFrameT = TypeVar("DataFrameT", pl.DataFrame, pl.LazyFrame)
SourceId = str | int
SourceT = TypeVar("SourceT")
P = ParamSpec("P")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
