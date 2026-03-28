from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.io.adapters.mds import MDSHeteroDataset

__all__ = ["MDSHeteroDataset"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "MDSHeteroDataset": ("dronalize.io.adapters.mds", "MDSHeteroDataset"),
}


def __getattr__(name: str) -> object:
    """Resolve optional dataset exports lazily."""
    if name not in _EXPORTS:
        msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    module_name, export_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), export_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy dataset exports during interactive discovery."""
    return sorted(set(globals()) | set(__all__))
