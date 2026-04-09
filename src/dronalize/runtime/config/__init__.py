"""Runtime configuration models and config-resolution helpers.

## Import guide

```python
from dronalize.runtime.config import ConfigFile, ConfigResolver, ResolvedConfig
from dronalize.runtime.config import load_project_config, resolve_runtime_config
```

This package groups the public configuration types used by the runtime planner.
It covers three related views of configuration:

- authoring-side file models such as
  [`ConfigFile`][dronalize.runtime.config.ConfigFile] and
  [`FileDatasetConfig`][dronalize.runtime.config.FileDatasetConfig]
- override helpers such as [`PlanOverrides`][dronalize.runtime.config.PlanOverrides]
- resolved runtime models such as
  [`ResolvedConfig`][dronalize.runtime.config.ResolvedConfig]

[`ConfigResolver`][dronalize.runtime.config.ConfigResolver] and
[`resolve_runtime_config`][dronalize.runtime.config.resolve_runtime_config] are
the main entry points when combining dataset defaults with file-loaded or
programmatic overrides.

## Related modules

- [`dronalize.runtime`][] for higher-level planning and run-state models
- [`dronalize.processing`][] for processing-side config models
"""

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
    """Resolve file-loaded config against an existing resolved runtime config.

    Parameters
    ----------
    default : ResolvedConfig
        Existing resolved runtime configuration, usually derived from dataset
        defaults and earlier override layers.
    overrides : FileDatasetConfig | None, optional
        File-shaped overrides to merge into ``default``.

    Returns
    -------
    ResolvedConfig
        Fully merged runtime configuration.
    """
    return ConfigResolver().resolve_from_defaults(default=default, overrides=overrides)
