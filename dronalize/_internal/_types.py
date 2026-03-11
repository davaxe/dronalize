from __future__ import annotations

from typing import ParamSpec, TypeVar

import polars as pl

DataFrameT = TypeVar("DataFrameT", pl.DataFrame, pl.LazyFrame)
SourceId = str | int
SourceT = TypeVar("SourceT")
PayloadT = TypeVar("PayloadT")
P = ParamSpec("P")
T = TypeVar("T")
