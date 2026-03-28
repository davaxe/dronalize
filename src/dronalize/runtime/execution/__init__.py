"""Public execution entry points for dataset preprocessing."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.runtime.execution.executor import Progress
    from dronalize.runtime.execution.runner import (
        DatasetJob,
        DatasetRun,
        ProcessingSummary,
        prepare_dataset,
    )

__all__ = [
    "DatasetJob",
    "DatasetRun",
    "ProcessingSummary",
    "Progress",
    "prepare_dataset",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DatasetJob": ("dronalize.runtime.execution.runner", "DatasetJob"),
    "DatasetRun": ("dronalize.runtime.execution.runner", "DatasetRun"),
    "ProcessingSummary": ("dronalize.runtime.execution.runner", "ProcessingSummary"),
    "Progress": ("dronalize.runtime.execution.executor", "Progress"),
    "prepare_dataset": ("dronalize.runtime.execution.runner", "prepare_dataset"),
}


def __getattr__(name: str) -> object:
    """Resolve execution exports lazily to avoid importing the runner eagerly."""
    if name not in _EXPORTS:
        msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    module_name, export_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), export_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public exports during interactive discovery."""
    return sorted(set(globals()) | set(__all__))
