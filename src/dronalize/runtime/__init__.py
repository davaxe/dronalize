"""Runtime orchestration, configuration assembly, and CLI support."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from dronalize.runtime.config import (
    Config,
    ConfigOverrides,
    ExecutionConfig,
    load_config_overrides,
    resolve_runtime_config,
)

if TYPE_CHECKING:
    from dronalize.runtime.execution.runner import (
        DatasetJob,
        DatasetRun,
        ProcessingSummary,
        prepare_dataset,
    )

__all__ = [
    "Config",
    "ConfigOverrides",
    "DatasetJob",
    "DatasetRun",
    "ExecutionConfig",
    "ProcessingSummary",
    "load_config_overrides",
    "prepare_dataset",
    "resolve_runtime_config",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DatasetJob": ("dronalize.runtime.execution.runner", "DatasetJob"),
    "DatasetRun": ("dronalize.runtime.execution.runner", "DatasetRun"),
    "ProcessingSummary": ("dronalize.runtime.execution.runner", "ProcessingSummary"),
    "prepare_dataset": ("dronalize.runtime.execution.runner", "prepare_dataset"),
}


def __getattr__(name: str) -> object:
    """Resolve execution-facing exports lazily."""
    if name not in _EXPORTS:
        msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    module_name, export_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), export_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose eager and lazy exports during interactive discovery."""
    return sorted(set(globals()) | set(__all__))
