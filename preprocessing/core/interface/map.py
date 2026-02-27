"""Common data structures and utilities for map parsing and processing.

This module re-exports all public symbols from the split sub-modules
for backward compatibility.
"""

from preprocessing.core.interface.graph_builder import (
    Edges,
    GraphBuilder,
    InterpolationStage,
    get_edges_from_adj_list,
    interpolate_position,
)
from preprocessing.core.interface.map_nodes import (
    IntIDBaseMapObject,
    IntIdBaseMapObject,
    IntIDNode,
)
from preprocessing.core.interface.map_protocols import (
    BaseEnum,
    BaseMapObject,
    BaseNode,
)
from preprocessing.core.map_graph import MapGraph

__all__ = [
    "BaseEnum",
    "BaseMapObject",
    "BaseNode",
    "Edges",
    "GraphBuilder",
    "IntIDBaseMapObject",
    "IntIDNode",
    "IntIdBaseMapObject",
    "InterpolationStage",
    "MapGraph",
    "get_edges_from_adj_list",
    "interpolate_position",
]
