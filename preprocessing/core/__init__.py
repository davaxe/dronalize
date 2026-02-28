"""Core package — domain models, protocols, graph building, and parallelism."""

# Domain models
# Graph building
from preprocessing.core.datatypes import (
    AgentCategory,
    EdgeType,
    Explicit,
    Implicit,
    Loaded,
    MapContext,
    MapGraph,
    Scene,
)
from preprocessing.core.graph import (
    Edges,
    GraphBuilder,
    InterpolationStage,
    IntIDBaseMapObject,
    IntIdBaseMapObject,
    IntIDNode,
    get_edges_from_adj_list,
    interpolate_position,
)

# Parallel execution
from preprocessing.core.parallel import (
    ParallelLoader,
    ParallelSceneLoader,
    ProgressBar,
)

# Protocols & loader ABCs
from preprocessing.core.protocols import (
    BaseEnum,
    BaseMapObject,
    BaseNode,
    BaseSceneLoader,
    FilteringConfig,
    LoaderConfig,
    Resampling,
    SceneLoader,
    WindowParams,
)

__all__ = [
    "AgentCategory",
    "BaseEnum",
    "BaseMapObject",
    "BaseNode",
    "BaseSceneLoader",
    "EdgeType",
    "Edges",
    "Explicit",
    "FilteringConfig",
    "GraphBuilder",
    "Implicit",
    "IntIDBaseMapObject",
    "IntIDNode",
    "IntIdBaseMapObject",
    "InterpolationStage",
    "Loaded",
    "LoaderConfig",
    "MapContext",
    "MapGraph",
    "ParallelLoader",
    "ParallelSceneLoader",
    "ProgressBar",
    "Resampling",
    "Scene",
    "SceneLoader",
    "WindowParams",
    "get_edges_from_adj_list",
    "interpolate_position",
]
