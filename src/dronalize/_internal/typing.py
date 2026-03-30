from __future__ import annotations

from typing import TYPE_CHECKING, ParamSpec, TypeVar

import numpy as np

if TYPE_CHECKING:
    import polars as pl

    DataFrameT = TypeVar("DataFrameT", pl.DataFrame, pl.LazyFrame)
else:
    DataFrameT = TypeVar("DataFrameT")

SourceId = str | int
SourceT = TypeVar("SourceT")
P = ParamSpec("P")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
FloatScalarT = TypeVar("FloatScalarT", np.float32, np.float64)
FloatDType = type[np.float32] | type[np.float64]
