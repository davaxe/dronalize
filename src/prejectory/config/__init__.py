"""Public configuration entry points used by the CLI and Python API.

This module re-exports the main configuration models and utilities used by
`prejectory`. The main entry points include:

- [`ProjectConfig`][prejectory.config.ProjectConfig] for resolved project
  configuration loaded from a TOML file such as `config.toml`
- [`RuntimeOverride`][prejectory.config.RuntimeOverride] for basic CLI- or
  programmatic overrides layered on top of a dataset config
- [`parse_config`][prejectory.config.parse_config] for discovering
  and validating project configuration from disk

"""

from prejectory.config.models import PredictionTaskConfig, RuntimeOverride
from prejectory.config.parse import ProjectConfig, parse_config

__all__ = ["PredictionTaskConfig", "ProjectConfig", "RuntimeOverride", "parse_config"]
