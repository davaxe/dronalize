"""Core package — domain models, protocols, graph building, and parallelism."""

# Domain models
# Graph building
# Pipeline
# Allow `from dronalize.core import transforms` as a module reference
from dronalize.core import pipeline_factories, transforms
from dronalize.core.datatypes import (
    AgentCategory,
    DatasetSplit,
    EdgeType,
    LoaderConfig,
    MapGraph,
    MapKey,
    MapResolver,
    Scene,
    SplitNotSupportedError,
    fixed_map,
    keyed_map,
    no_map,
    preloaded_map,
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
    "DatasetSplit",
    "EdgeType",
    "Edges",
    "FlatMapTransform",
    "GraphBuilder",
    "InterpolationStage",
    "LoaderConfig",
    "MapGraph",
    "MapKey",
    "MapResolver",
    "ParallelSceneLoader",
    "Pipeline",
    "Point",
    "ProgressBar",
    "Scene",
    "SplitNotSupportedError",
    "Transform",
    "fixed_map",
    "get_edges_from_adj_list",
    "interpolate_position",
    "keyed_map",
    "no_map",
    "pipeline_factories",
    "preloaded_map",
    "transforms",
]
