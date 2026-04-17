"""Public configuration entry points used by the CLI and Python API.

This package exposes the smallest configuration surface most callers need:

- [`ProcessingConfig`][dronalize.config.ProcessingConfig] for resolved project
  configuration loaded from `pyproject.toml`
- [`RuntimeOverride`][dronalize.config.RuntimeOverride] for CLI- or
  programmatic overrides layered on top of a dataset config
- [`load_project_config`][dronalize.config.load_project_config] for discovering
  and validating project configuration from disk

Lower-level typed configuration models remain available from
[`dronalize.config.models`][] when a caller needs to construct or validate
individual config blocks directly.
"""

from dronalize.config.file import ProcessingConfig
from dronalize.config.reader import load_project_config
from dronalize.config.runtime import RuntimeOverride

__all__ = ["ProcessingConfig", "RuntimeOverride", "load_project_config"]
