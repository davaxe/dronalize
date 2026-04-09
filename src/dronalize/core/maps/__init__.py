"""Map graph types shared across loading, processing, and visualization.

## Import guide

```python
from dronalize.core.maps import MapGraph, SharedMapGraph, EdgeType
```

Use this package when you need the graph representation itself rather than the
map-processing helpers that build or resolve graphs at runtime.

- [`MapGraph`][dronalize.core.maps.MapGraph] stores the in-memory graph arrays
- [`SharedMapGraph`][dronalize.core.maps.SharedMapGraph] opens a graph backed by
  shared memory
- [`EdgeType`][dronalize.core.maps.EdgeType] provides the public edge-type enum
  used across builders and plots

## Related modules

- [`dronalize.processing.maps`][] for map builders, config, and resolver helpers
- [`dronalize.plot`][] for optional plotting helpers that consume
  [`MapGraph`][dronalize.core.maps.MapGraph]
"""

from dronalize.core.maps.edge_types import EdgeType
from dronalize.core.maps.graph import MapGraph, SharedMapGraph

__all__ = ["EdgeType", "MapGraph", "SharedMapGraph"]
