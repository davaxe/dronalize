"""Core package — domain models, protocols, graph building, and parallelism."""

# Domain models
# Graph building
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
    # Models
    "AgentCategory",
    "EdgeType",
    "Explicit",
    "Implicit",
    "Loaded",
    "MapContext",
    "MapGraph",
    "Scene",
    # Protocols & loader ABCs
    "BaseEnum",
    "BaseMapObject",
    "BaseNode",
    "BaseSceneLoader",
    "FilteringConfig",
    "LoaderConfig",
    "Resampling",
    "SceneLoader",
    "WindowParams",
    # Graph
    "Edges",
    "GraphBuilder",
    "IntIDBaseMapObject",
    "IntIDNode",
    "IntIdBaseMapObject",
    "InterpolationStage",
    "get_edges_from_adj_list",
    "interpolate_position",
    # Parallel
    "ParallelLoader",
    "ParallelSceneLoader",
    "ProgressBar",
]
