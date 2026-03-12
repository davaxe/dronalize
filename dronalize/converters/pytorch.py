from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dronalize._internal._compat import require_optional

if TYPE_CHECKING:
    from dronalize.maps.graph import MapGraph


def map_graph_to_torch(graph: MapGraph) -> dict[Any, Any]:
    """Convert the `MapGraph` to a format compatible with PyTorch Geometric.

    Uses `torch.from_numpy` for zero-copy conversion (the returned tensors
    share the same underlying memory as the NumPy arrays).

    Parameters
    ----------
    graph : MapGraph
        The MapGraph to convert.

    Returns
    -------
    dict
        Dictionary with node and edge data suitable for PyTorch Geometric.

    """
    torch = require_optional("torch", extra="torch")

    return {
        "map_point": {
            "num_nodes": graph.num_nodes,
            "type": torch.from_numpy(graph.node_types),
            "position": torch.from_numpy(graph.node_positions),
        },
        ("map_point", "to", "map_point"): {
            "edge_index": torch.from_numpy(graph.edge_indices),
            "type": torch.from_numpy(graph.edge_types),
        },
    }
