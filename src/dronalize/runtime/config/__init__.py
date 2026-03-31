"""Public runtime configuration API."""

from __future__ import annotations

from dronalize.runtime.config.file import (
    ConfigFile,
    FileDatasetConfig,
    FileExecutionConfig,
    load_project_config,
)
from dronalize.runtime.config.overrides import PlanOverrides
from dronalize.runtime.config.resolved import ResolvedConfig, ResolvedExecutionConfig
from dronalize.runtime.config.resolver import ConfigResolver

__all__ = [
    "ConfigFile",
    "ConfigResolver",
    "FileDatasetConfig",
    "FileExecutionConfig",
    "PlanOverrides",
    "ResolvedConfig",
    "ResolvedExecutionConfig",
    "load_project_config",
    "resolve_runtime_config",
]


def resolve_runtime_config(
    *, default: ResolvedConfig, overrides: FileDatasetConfig | None = None
) -> ResolvedConfig:
    """Resolve file-loaded config against an existing resolved runtime config."""
    return ConfigResolver().resolve_from_defaults(default=default, overrides=overrides)
