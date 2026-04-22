"""Public configuration entry points used by he CLI and Python API.

This package exposes the smallest configuration surface most callers need:

- [`ProcessingConfig`][dronalize.config.ProcessingConfig] for resolved project
  configuration loaded from a TOML file such as `config.toml`
- [`RuntimeOverride`][dronalize.config.RuntimeOverride] for basic CLI- or
  programmatic overrides layered on top of a dataset config
- [`parse_config`][dronalize.config.parse_config] for discovering
  and validating project configuration from disk

Lower-level typed configuration models remain available from
[`dronalize.config.models`][] when a caller needs to construct or validate
individual config blocks directly.
"""

from dronalize.config.file import ProcessingConfig
from dronalize.config.reader import parse_config
from dronalize.config.runtime import RuntimeOverride

__all__ = ["ProcessingConfig", "RuntimeOverride", "parse_config"]
