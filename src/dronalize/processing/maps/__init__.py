"""Public map-processing API.

## Import guide

```python
from dronalize.processing.maps import BaseMapBuilder, shared_map
from dronalize.config.models import MapConfig
```

This package groups the map-specific pieces that are shared across loaders and
runtime processing:

- [`MapConfig`][dronalize.config.models.MapConfig] and extraction models
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

from dronalize.processing.maps.builder import BaseMapBuilder, MapBuilder
from dronalize.processing.maps.resolver import no_map, shared_map

__all__ = ["BaseMapBuilder", "MapBuilder", "no_map", "shared_map"]
