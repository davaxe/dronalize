"""Small root package for shared enums used across the public API.

## Import guide

```python
from dronalize.core import AgentCategory, DatasetSplit
from dronalize.core.scene import Scene, TrajectorySchema
from dronalize.core.maps import MapGraph, EdgeType
```

The richer domain-model surfaces live in sibling packages:

- [`dronalize.core.scene`][] for scenes, trajectory schemas, and schema lookup
- [`dronalize.core.maps`][] for map graphs and map-edge types

This root package intentionally only exposes the cross-cutting enum types that
appear throughout the rest of the library.

"""

from dronalize.core.categories import (
    AgentCategory,
    AgentCategoryInput,
    AgentCategoryLike,
    DatasetSplit,
)

__all__ = ["AgentCategory", "AgentCategoryInput", "AgentCategoryLike", "DatasetSplit"]
