from __future__ import annotations

from collections.abc import Hashable
from typing import ParamSpec, TypeVar

import polars as pl

DataFrameT = TypeVar("DataFrameT", pl.DataFrame, pl.LazyFrame)
SourceT = TypeVar("SourceT")
SourceT_co = TypeVar("SourceT_co", covariant=True)
IdT = TypeVar("IdT", bound=Hashable)
PayloadT = TypeVar("PayloadT")
P = ParamSpec("P")
T = TypeVar("T")
