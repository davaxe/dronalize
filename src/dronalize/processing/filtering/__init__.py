"""Public filtering surface grouped by rule family.

## Import guide

```python
from dronalize.processing import filtering
from dronalize.processing.filtering import Filter, AgentSelector, Tolerance, tol
```

This package is organized around four kinds of public symbols:

- root-level filter containers such as
  [`Filter`][dronalize.processing.filtering.Filter] and
  [`FilterSpec`][dronalize.processing.filtering.FilterSpec]
- helper functions such as
  [`filter_scene`][dronalize.processing.filtering.filter_scene] and
  [`merge_filters`][dronalize.processing.filtering.merge_filters]
- tolerance models for agent-rule aggregation
- grouped rule families exposed through the
  [`cleanup`][dronalize.processing.filtering.cleanup],
  [`scene`][dronalize.processing.filtering.scene], and
  [`agent`][dronalize.processing.filtering.agent] submodules

This keeps the common filtering API discoverable without flattening every rule
type into one package namespace.

## Related modules

- [`dronalize.processing`][] for higher-level processing config
- [`dronalize.processing.filtering.agent`][] for agent-rule definitions
- [`dronalize.processing.filtering.tolerance`][] for tolerance models
"""

from dronalize.processing.filtering import agent, cleanup, scene
from dronalize.processing.filtering.apply import filter_scene
from dronalize.processing.filtering.context import AgentSelector
from dronalize.processing.filtering.filter import Filter, FilterSpec, merge_filters
from dronalize.processing.filtering.tolerance import (
    AbsoluteTolerance,
    CombinedTolerance,
    RelativeTolerance,
    Tolerance,
    tol,
)

__all__ = [
    "AbsoluteTolerance",
    "AgentSelector",
    "CombinedTolerance",
    "Filter",
    "FilterSpec",
    "RelativeTolerance",
    "Tolerance",
    "agent",
    "cleanup",
    "filter_scene",
    "merge_filters",
    "scene",
    "tol",
]
