"""Grouped processing-facing public API.

## Import guide

```python
from dronalize.processing import LoaderConfig, WindowConfig, MapConfig
from dronalize.processing import filtering, maps, pipeline
```

Use this package for the processing config models that most callers need
directly, and then drop into the focused subpackages for the richer APIs:

- [`filtering`][dronalize.processing.filtering] for filter containers and
  grouped rule families
- [`maps`][dronalize.processing.maps] for map builders, extraction config, and
  resolver helpers
- [`pipeline`][dronalize.processing.pipeline] for pipeline and transform
  abstractions

The advanced custom dataset loader API lives one level deeper in
[`dronalize.processing.loading`][] instead of being flattened into this root
surface.

## Related modules

- [`dronalize.processing.loading`][] for the advanced loader extension API
- [`dronalize.runtime`][] for runtime config resolution and planning
"""

from dronalize.processing import filtering, maps, pipeline
from dronalize.processing.loading import LaneChangeSamplingConfig, LoaderConfig, WindowConfig
from dronalize.processing.maps import MapConfig

__all__ = [
    "LaneChangeSamplingConfig",
    "LoaderConfig",
    "MapConfig",
    "WindowConfig",
    "filtering",
    "maps",
    "pipeline",
]
