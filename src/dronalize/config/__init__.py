"""Public configuration entry points used by the CLI and Python API.

This module re-exports the main configuration models and utilities used by
`dronalize`. The main entry points include:

- [`ProjectConfig`][dronalize.config.ProjectConfig] for resolved project
  configuration loaded from a TOML file such as `config.toml`
- [`RuntimeOverride`][dronalize.config.RuntimeOverride] for basic CLI- or
  programmatic overrides layered on top of a dataset config
- [`parse_config`][dronalize.config.parse_config] for discovering
  and validating project configuration from disk

"""

from dronalize.config.parse import parse_config
from dronalize.config.project import ProjectConfig
from dronalize.config.runtime import RuntimeOverride

__all__ = ["ProjectConfig", "RuntimeOverride", "parse_config"]
