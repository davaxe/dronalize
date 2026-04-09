"""Public map-processing API.

## Import guide

```python
from dronalize.processing.maps import MapConfig, BaseMapBuilder, shared_map
```

This package groups the map-specific pieces that are shared across loaders and
runtime processing:

- [`MapConfig`][dronalize.processing.maps.MapConfig] and extraction models
  describe how much map context to keep
- [`BaseMapBuilder`][dronalize.processing.maps.BaseMapBuilder] and
  [`MapBuilder`][dronalize.processing.maps.MapBuilder] define the graph-building
  contract
- [`no_map`][dronalize.processing.maps.no_map] and
  [`shared_map`][dronalize.processing.maps.shared_map] provide runtime map
  resolver helpers

The graph data structures themselves live in [`dronalize.core.maps`][].

## Related modules

- [`dronalize.core.maps`][] for map graph containers and edge types
- [`dronalize.processing.loading`][] for loader APIs that attach map resolvers
"""

from dronalize.processing.maps.builder import BaseMapBuilder, MapBuilder, Point
from dronalize.processing.maps.config import (
    BoundingBoxExtraction,
    CircularExtraction,
    FullMapExtraction,
    MapConfig,
    SceneExtentExtraction,
)
from dronalize.processing.maps.resolver import no_map, shared_map

__all__ = [
    "BaseMapBuilder",
    "BoundingBoxExtraction",
    "CircularExtraction",
    "FullMapExtraction",
    "MapBuilder",
    "MapConfig",
    "Point",
    "SceneExtentExtraction",
    "no_map",
    "shared_map",
]
