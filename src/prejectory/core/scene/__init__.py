"""Scene models, trajectory schemas, and schema lookup helpers.

## Import guide

``python
from prejectory.core.scene import Scene, TrajectorySchema, TrajectoryField
from prejectory.core.scene import CANONICAL, get_trajectory_schema
``

This package is the main home for scene-facing domain types:

- [`Scene`][prejectory.core.scene.Scene] is the normalized scene container used
  throughout processing
- [`TrajectorySchema`][prejectory.core.scene.TrajectorySchema] and
  [`TrajectoryField`][prejectory.core.scene.TrajectoryField] describe schema
  shape and fields
- built-in schema constants such as [`CANONICAL`][prejectory.core.scene.CANONICAL]
  and [`POSITIONS_ONLY`][prejectory.core.scene.POSITIONS_ONLY] provide
  stable built-in schema variants
- helper functions expose registered schemas and resolve schema-like inputs

The symbols exported here are intended to be imported directly by code that
works with scenes or persisted feature layouts.

## Related modules

- [`prejectory.core.maps`][] for map graph types referenced by map resolvers
- [`prejectory.io`][] for export-side configuration and persisted storage
  contracts
"""

from prejectory.core.scene.model import MapResolver, Scene
from prejectory.core.scene.schema import (
    CANONICAL,
    POSITIONS_ONLY,
    POSITIONS_VELOCITY,
    POSITIONS_VELOCITY_ACCELERATION,
    POSITIONS_VELOCITY_YAW,
    POSITIONS_YAW,
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
    "MapResolver",
    "Scene",
    "TrajectoryField",
    "TrajectorySchema",
    "available_trajectory_schema_names",
    "available_trajectory_schemas",
    "get_trajectory_schema",
]
