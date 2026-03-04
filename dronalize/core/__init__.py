"""Core package — domain models, protocols, graph building, and parallelism."""

# Domain models
# Graph building
# Pipeline
# Allow `from dronalize.core import transforms` as a module reference
from dronalize.core import transforms
from dronalize.core.datatypes import (
    AgentCategory,
    EdgeType,
    Explicit,
    Implicit,
    Loaded,
    LoaderConfig,
    MapContext,
    MapGraph,
    Scene,
)
from dronalize.core.graph import (
    Edges,
    GraphBuilder,
    InterpolationStage,
    Point,
    get_edges_from_adj_list,
    interpolate_position,
)

# Parallel execution
from dronalize.core.parallel import (
    ParallelSceneLoader,
    ProgressBar,
)
from dronalize.core.pipeline import (
    FlatMapTransform,
    Pipeline,
    Transform,
)

# Protocols & loader ABCs
from dronalize.core.protocols import (
    BaseEnum,
    BaseMapObject,
    BaseSceneLoader,
)

__all__ = [
    "AgentCategory",
    "BaseEnum",
    "BaseMapObject",
    "BaseSceneLoader",
    "EdgeType",
    "Edges",
    "Explicit",
    "FlatMapTransform",
    "GraphBuilder",
    "Implicit",
    "InterpolationStage",
    "Loaded",
    "LoaderConfig",
    "MapContext",
    "MapGraph",
    "ParallelSceneLoader",
    "Pipeline",
    "Point",
    "ProgressBar",
    "Scene",
    "Transform",
    "get_edges_from_adj_list",
    "interpolate_position",
    "transforms",
]
