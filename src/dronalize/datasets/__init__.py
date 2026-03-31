"""Built-in dataset registry and lazy dataset package exports."""

from __future__ import annotations

import importlib
import pkgutil

from dronalize.datasets.registry import (
    DatasetCapabilities,
    DatasetDescriptor,
    ExecutionScope,
    available,
    get,
    register,
)

__all__ = [
    "DatasetCapabilities",
    "DatasetDescriptor",
    "ExecutionScope",
    "available",
    "get",
    "register",
]

_DATASET_MODULES = {
    module_info.name
    for module_info in pkgutil.iter_modules(__path__)
    if module_info.name not in {"registry", "shared"}
}


def __getattr__(name: str) -> object:
    """Lazily expose dataset submodules from the package root."""
    if name in _DATASET_MODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Expose lazy dataset modules during interactive discovery."""
    return sorted(set(globals()) | _DATASET_MODULES)
