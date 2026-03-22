"""Curated public API for the dronalize library."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.categories import AgentCategory, DatasetSplit
    from dronalize.config.config import (
        Config,
        ConfigOverrides,
        ExecutionConfig,
        load_config_overrides,
        resolve_runtime_config,
    )
    from dronalize.config.loader import LoaderConfig, WindowParams
    from dronalize.config.map import MapConfig
    from dronalize.config.writer import MDSFormatConfig, WriterConfig
    from dronalize.execution.runner import DatasetJob, DatasetRun, prepare_dataset
    from dronalize.scene._scene import Scene

__all__ = [
    "AgentCategory",
    "Config",
    "ConfigOverrides",
    "DatasetJob",
    "DatasetRun",
    "DatasetSplit",
    "ExecutionConfig",
    "LoaderConfig",
    "MDSFormatConfig",
    "MapConfig",
    "Scene",
    "WindowParams",
    "WriterConfig",
    "load_config_overrides",
    "prepare_dataset",
    "resolve_runtime_config",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentCategory": ("dronalize.categories", "AgentCategory"),
    "Config": ("dronalize.config.config", "Config"),
    "ConfigOverrides": ("dronalize.config.config", "ConfigOverrides"),
    "DatasetJob": ("dronalize.execution.runner", "DatasetJob"),
    "DatasetRun": ("dronalize.execution.runner", "DatasetRun"),
    "DatasetSplit": ("dronalize.categories", "DatasetSplit"),
    "ExecutionConfig": ("dronalize.config.config", "ExecutionConfig"),
    "LoaderConfig": ("dronalize.config.loader", "LoaderConfig"),
    "MDSFormatConfig": ("dronalize.config.writer", "MDSFormatConfig"),
    "MapConfig": ("dronalize.config.map", "MapConfig"),
    "Scene": ("dronalize.scene._scene", "Scene"),
    "WindowParams": ("dronalize.config.loader", "WindowParams"),
    "WriterConfig": ("dronalize.config.writer", "WriterConfig"),
    "load_config_overrides": ("dronalize.config.config", "load_config_overrides"),
    "prepare_dataset": ("dronalize.execution.runner", "prepare_dataset"),
    "resolve_runtime_config": ("dronalize.config.config", "resolve_runtime_config"),
}


def __getattr__(name: str) -> object:
    """Resolve public API symbols lazily to keep package imports lightweight."""
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
