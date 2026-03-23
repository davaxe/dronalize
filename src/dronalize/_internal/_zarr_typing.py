"""Internal utilities for type checking zarr objects."""

from __future__ import annotations

from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
import zarr

DType = TypeVar("DType", bound=np.generic)


def require_group(parent: zarr.Group, key: str) -> zarr.Group:
    obj = parent[key]
    if not isinstance(obj, zarr.Group):
        msg = f"{key!r} is not a zarr.Group"
        raise TypeError(msg)
    return obj


def require_array(parent: zarr.Group, key: str) -> zarr.Array[Any]:
    obj = parent[key]
    if not isinstance(obj, zarr.Array):
        msg = f"{key!r} is not a zarr.Array"
        raise TypeError(msg)
    return obj


def read_ndarray(
    parent: zarr.Group,
    key: str,
) -> npt.NDArray[np.generic]:
    arr = require_array(parent, key)
    data = arr[:]
    if not isinstance(data, np.ndarray):
        msg = f"{key!r} did not produce a NumPy ndarray"
        raise TypeError(msg)
    return data


def read_typed_array(
    parent: zarr.Group,
    key: str,
    dtype: type[DType],
) -> npt.NDArray[DType]:
    data = read_ndarray(parent, key)
    if data.dtype != np.dtype(dtype):
        msg = f"{key!r} has dtype {data.dtype}, expected {np.dtype(dtype)}"
        raise TypeError(msg)
    return cast("npt.NDArray[DType]", data)
