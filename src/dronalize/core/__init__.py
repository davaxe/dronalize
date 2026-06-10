"""Core scene, map, and shared enum types.

## Import guide

```python
from dronalize.core import AgentCategory, DatasetSplit, MapGraph, Scene
from dronalize.core import CANONICAL, TrajectorySchema, get_trajectory_schema
from dronalize.core import functional
```

This package is the public import surface for the common domain objects used by
readers, runtime planning, and downstream model code.

"""

from dronalize.core.categories import (
    AgentCategory,
    AgentCategoryInput,
    AgentCategoryLike,
    DatasetSplit,
    EdgeType,
)
from dronalize.core.maps import MapGraph, SharedMapGraph
from dronalize.core.scene import (
    CANONICAL,
    POSITIONS_ONLY,
    POSITIONS_VELOCITY,
    POSITIONS_VELOCITY_ACCELERATION,
    POSITIONS_VELOCITY_YAW,
    POSITIONS_YAW,
    TRAJECTORY_SCHEMAS,
    MapResolver,
    Scene,
    TrajectoryField,
    TrajectorySchema,
    available_trajectory_schema_names,
    available_trajectory_schemas,
    get_trajectory_schema,
)

__all__ = [
    "CANONICAL",
    "POSITIONS_ONLY",
    "POSITIONS_VELOCITY",
    "POSITIONS_VELOCITY_ACCELERATION",
    "POSITIONS_VELOCITY_YAW",
    "POSITIONS_YAW",
    "TRAJECTORY_SCHEMAS",
    "AgentCategory",
    "AgentCategoryInput",
    "AgentCategoryLike",
    "DatasetSplit",
    "EdgeType",
    "MapGraph",
    "MapResolver",
    "Scene",
    "SharedMapGraph",
    "TrajectoryField",
    "TrajectorySchema",
    "available_trajectory_schema_names",
    "available_trajectory_schemas",
    "get_trajectory_schema",
]
