from __future__ import annotations

from typing import ParamSpec, TypeVar

import polars as pl

DataFrameT = TypeVar("DataFrameT", pl.DataFrame, pl.LazyFrame)
SceneId = str | int
SourceT = TypeVar("SourceT")
SourceT_co = TypeVar("SourceT_co", covariant=True)
PayloadT = TypeVar("PayloadT")
P = ParamSpec("P")
T = TypeVar("T")
