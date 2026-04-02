from __future__ import annotations

from pathlib import Path
from typing import ParamSpec

import numpy as np
import polars as pl
from typing_extensions import TypeVar

DataFrameT = TypeVar("DataFrameT", pl.DataFrame, pl.LazyFrame)

SourceId = str | int
SourceT = TypeVar("SourceT", default=Path)
P = ParamSpec("P")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
FloatScalarT = TypeVar("FloatScalarT", np.float32, np.float64)
FloatDType = type[np.float32] | type[np.float64]
