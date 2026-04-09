"""Scene models, trajectory schemas, and schema lookup helpers.

## Import guide

```python
from dronalize.core.scene import Scene, TrajectorySchema, TrajectoryField
from dronalize.core.scene import CANONICAL, get_trajectory_schema
```

This package is the main home for scene-facing domain types:

- [`Scene`][dronalize.core.scene.Scene] is the normalized scene container used
  throughout processing
- [`TrajectorySchema`][dronalize.core.scene.TrajectorySchema] and
  [`TrajectoryField`][dronalize.core.scene.TrajectoryField] describe schema
  shape and fields
- built-in schema constants such as [`CANONICAL`][dronalize.core.scene.CANONICAL]
  and [`POSITIONS_ONLY`][dronalize.core.scene.POSITIONS_ONLY] provide
  stable predefined schema variants
- helper functions expose registered schemas and resolve schema-like inputs

The symbols exported here are intended to be imported directly by code that
works with scenes or persisted feature layouts.

## Related modules

- [`dronalize.core.maps`][] for map graph types referenced by map resolvers
- [`dronalize.io`][] for export-side configuration and persisted storage
  contracts
"""

from dronalize.core.scene.model import MapKey, MapResolver, Scene
from dronalize.core.scene.schema import (
    CANONICAL,
    POSITIONS_ONLY,
    POSITIONS_VELOCITY,
    POSITIONS_VELOCITY_ACCELERATION,
    POSITIONS_VELOCITY_YAW,
    POSITIONS_YAW,
    TRAJECTORY_SCHEMAS,
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
    "MapKey",
    "MapResolver",
    "Scene",
    "TrajectoryField",
    "TrajectorySchema",
    "available_trajectory_schema_names",
    "available_trajectory_schemas",
    "get_trajectory_schema",
]
