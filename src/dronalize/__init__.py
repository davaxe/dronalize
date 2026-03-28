"""Curated public API for the dronalize library."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.core.categories import AgentCategory, DatasetSplit
    from dronalize.core.scene.model import Scene
    from dronalize.io.config import MDSFormatConfig, WriterConfig
    from dronalize.processing.ingest.config import LoaderConfig, WindowParams
    from dronalize.processing.maps.config import MapConfig
    from dronalize.runtime.config import (
        Config,
        ConfigOverrides,
        ExecutionConfig,
        load_config_overrides,
        resolve_runtime_config,
    )
    from dronalize.runtime.execution.runner import DatasetJob, DatasetRun, prepare_dataset

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
    "AgentCategory": ("dronalize.core.categories", "AgentCategory"),
    "Config": ("dronalize.runtime.config", "Config"),
    "ConfigOverrides": ("dronalize.runtime.config", "ConfigOverrides"),
    "DatasetJob": ("dronalize.runtime.execution.runner", "DatasetJob"),
    "DatasetRun": ("dronalize.runtime.execution.runner", "DatasetRun"),
    "DatasetSplit": ("dronalize.core.categories", "DatasetSplit"),
    "ExecutionConfig": ("dronalize.runtime.config", "ExecutionConfig"),
    "LoaderConfig": ("dronalize.processing.ingest.config", "LoaderConfig"),
    "MDSFormatConfig": ("dronalize.io.config", "MDSFormatConfig"),
    "MapConfig": ("dronalize.processing.maps.config", "MapConfig"),
    "Scene": ("dronalize.core.scene.model", "Scene"),
    "WindowParams": ("dronalize.processing.ingest.config", "WindowParams"),
    "WriterConfig": ("dronalize.io.config", "WriterConfig"),
    "load_config_overrides": ("dronalize.runtime.config", "load_config_overrides"),
    "prepare_dataset": ("dronalize.runtime.execution.runner", "prepare_dataset"),
    "resolve_runtime_config": ("dronalize.runtime.config", "resolve_runtime_config"),
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
