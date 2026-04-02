"""Curated public API for the dronalize library."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.core.categories import AgentCategory, DatasetSplit
    from dronalize.core.scene.model import Scene
    from dronalize.io.config import MDSFormatConfig, WriterConfig
    from dronalize.processing.ingest.config import LoaderConfig, WindowConfig
    from dronalize.processing.maps.config import MapConfig
    from dronalize.runtime.config import (
        ConfigFile,
        ConfigResolver,
        FileDatasetConfig,
        FileExecutionConfig,
        PlanOverrides,
        ResolvedConfig,
        ResolvedExecutionConfig,
        load_project_config,
        resolve_runtime_config,
    )
    from dronalize.runtime.models import DatasetPlan, DatasetRun
    from dronalize.runtime.planning import plan_dataset

__all__ = [
    "AgentCategory",
    "ConfigFile",
    "ConfigResolver",
    "DatasetPlan",
    "DatasetRun",
    "DatasetSplit",
    "FileDatasetConfig",
    "FileExecutionConfig",
    "LoaderConfig",
    "MDSFormatConfig",
    "MapConfig",
    "PlanOverrides",
    "ResolvedConfig",
    "ResolvedExecutionConfig",
    "Scene",
    "WindowConfig",
    "WriterConfig",
    "load_project_config",
    "plan_dataset",
    "resolve_runtime_config",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentCategory": ("dronalize.core.categories", "AgentCategory"),
    "ConfigResolver": ("dronalize.runtime.config", "ConfigResolver"),
    "DatasetPlan": ("dronalize.runtime.models", "DatasetPlan"),
    "DatasetRun": ("dronalize.runtime.models", "DatasetRun"),
    "DatasetSplit": ("dronalize.core.categories", "DatasetSplit"),
    "FileDatasetConfig": ("dronalize.runtime.config", "FileDatasetConfig"),
    "FileExecutionConfig": ("dronalize.runtime.config", "FileExecutionConfig"),
    "LoaderConfig": ("dronalize.processing.ingest.config", "LoaderConfig"),
    "MDSFormatConfig": ("dronalize.io.config", "MDSFormatConfig"),
    "MapConfig": ("dronalize.processing.maps.config", "MapConfig"),
    "PlanOverrides": ("dronalize.runtime.config", "PlanOverrides"),
    "ResolvedConfig": ("dronalize.runtime.config", "ResolvedConfig"),
    "ResolvedExecutionConfig": ("dronalize.runtime.config", "ResolvedExecutionConfig"),
    "Scene": ("dronalize.core.scene.model", "Scene"),
    "WindowConfig": ("dronalize.processing.ingest.config", "WindowConfig"),
    "WriterConfig": ("dronalize.io.config", "WriterConfig"),
    "load_project_config": ("dronalize.runtime.config", "load_project_config"),
    "plan_dataset": ("dronalize.runtime.planning", "plan_dataset"),
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
