"""Runtime planning, configuration assembly, and run-state models.

## Import guide

```python
from dronalize.runtime import plan_dataset, summarize_plan
from dronalize.runtime import ConfigResolver, ResolvedConfig
```

This package groups the public runtime API used by Python callers:

- config authoring and resolved config models
- config loading and resolution helpers
- prepared planning and live run models
- summary helpers for prepared runs

Low-level executor protocols and concrete executors still exist, but they are
documented as advanced runtime hooks in their dedicated modules rather than
being part of this root package surface.

## Related modules

- [`dronalize.runtime.config`][] for configuration models and config-resolution
  helpers
- [`dronalize.runtime.executor`][dronalize.runtime.executor] for advanced
  executor protocols and progress models
- [`dronalize.datasets`][] for dataset descriptors consumed during planning
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from dronalize.runtime.models import DatasetPlan, DatasetRun, ProcessingSummary
    from dronalize.runtime.planning import plan_dataset
    from dronalize.runtime.summary import summarize_plan

__all__ = [
    "ConfigFile",
    "ConfigResolver",
    "DatasetPlan",
    "DatasetRun",
    "FileDatasetConfig",
    "FileExecutionConfig",
    "PlanOverrides",
    "ProcessingSummary",
    "ResolvedConfig",
    "ResolvedExecutionConfig",
    "load_project_config",
    "plan_dataset",
    "resolve_runtime_config",
    "summarize_plan",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DatasetPlan": ("dronalize.runtime.models", "DatasetPlan"),
    "DatasetRun": ("dronalize.runtime.models", "DatasetRun"),
    "ProcessingSummary": ("dronalize.runtime.models", "ProcessingSummary"),
    "plan_dataset": ("dronalize.runtime.planning", "plan_dataset"),
    "summarize_plan": ("dronalize.runtime.summary", "summarize_plan"),
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
