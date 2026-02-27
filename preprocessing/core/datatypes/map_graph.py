from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from preprocessing.core._compat import require_optional

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(init=False, repr=False)
class MapGraph:
    """Represents a graph structure for a map, including nodes and edges.

    All internal arrays use NumPy so that the core library does not depend on
    PyTorch.  Conversion to `torch.Tensor` is available via
    `to_torch_graph` (zero-copy when possible through
    `torch.from_numpy`).
    """

    node_positions: npt.NDArray[np.float32]
    """Node positions, shape `(N, 2)`."""

    edge_indices: npt.NDArray[np.int64]
    """Edge COO indices, shape `(2, M)`."""

    node_types: npt.NDArray[np.int64]
    """Per-node type labels, shape `(N,)`."""

    edge_types: npt.NDArray[np.int64]
    """Per-edge type labels, shape `(M,)`."""

    num_nodes: int
    """Number of nodes in the graph."""

    num_edges: int
    """Number of edges in the graph."""

    def __init__(
        self,
        node_positions: npt.NDArray[np.float32],
        edge_indices: npt.NDArray[np.int64],
        node_types: npt.NDArray[np.int64] | None = None,
        edge_types: npt.NDArray[np.int64] | None = None,
        *,
        return_if_empty: bool = True,
    ) -> None:
        """Initialize a `MapGraph` instance.

        Args:
            node_positions: array of shape `(N, 2)` with node positions.
            edge_indices: array of shape `(2, M)` with edge endpoint indices.
            node_types: array of shape `(N,)` with node type labels. Defaults to `None`, which means
                all nodes are of type 1.
            edge_types: array of shape `(M,)` with edge type labels. Defaults to `None`, which means
                all edges are of type 1.
            return_if_empty: if `True`, allows empty graphs and returns empty arrays.  If `False`,
                raises `ValueError` when both `node_positions` and `edge_indices` are empty.

        """
        if node_positions.size == 0 and edge_indices.size == 0:
            if return_if_empty:
                node_positions = np.zeros((0, 2), dtype=np.float32)
                edge_indices = np.zeros((2, 0), dtype=np.int64)
                node_types = np.zeros((0,), dtype=np.int64)
                edge_types = np.zeros((0,), dtype=np.int64)
            else:
                msg = (
                    "Node positions and edge indices cannot be empty."
                    " Set `return_if_empty=True` to allow empty graphs."
                )
                raise ValueError(msg)

        self.node_positions = np.ascontiguousarray(node_positions, dtype=np.float32)
        self.edge_indices = np.ascontiguousarray(edge_indices, dtype=np.int64)
        self.num_nodes = int(node_positions.shape[0])
        self.num_edges = int(edge_indices.shape[1]) if edge_indices.size > 0 else 0

        self.node_types = (
            np.ascontiguousarray(node_types, dtype=np.int64)
            if node_types is not None
            else np.ones(self.num_nodes, dtype=np.int64)
        )

        self.edge_types = (
            np.ascontiguousarray(edge_types, dtype=np.int64)
            if edge_types is not None
            else np.ones(self.num_edges, dtype=np.int64)
        )

    # ------------------------------------------------------------------
    # Torch conversion (lazy import - zero-copy when possible)
    # ------------------------------------------------------------------

    def to_torch_graph(self) -> dict[Any, Any]:
        """Convert the `MapGraph` to a format compatible with PyTorch Geometric.

        Uses `torch.from_numpy` for zero-copy conversion (the returned tensors share the same
        underlying memory as the NumPy arrays).

        Returns:
            Dictionary with node and edge data suitable for PyTorch Geometric.

        """
        torch = require_optional("torch", extra="torch")

        return {
            "map_point": {
                "num_nodes": self.num_nodes,
                "type": torch.from_numpy(self.node_types),
                "position": torch.from_numpy(self.node_positions),
            },
            ("map_point", "to", "map_point"): {
                "edge_index": torch.from_numpy(self.edge_indices),
                "type": torch.from_numpy(self.edge_types),
            },
        }

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def extract_radius(
        self,
        center: tuple[float, float] | npt.NDArray[np.floating],
        radius: float,
    ) -> MapGraph:
        """Extract a subgraph within a specified radius from a center point.

        Args:
            center: center of the radius as a tuple `(x, y)` or array.
            radius: radius within which to extract nodes and edges.

        Returns:
            A new `MapGraph` containing only the nodes and edges within the
            given radius.

        """
        center_arr = np.asarray(center, dtype=np.float32)

        # Squared-distance avoids the sqrt
        diff = self.node_positions - center_arr
        dist_sq = np.sum(diff * diff, axis=1)
        within_mask: npt.NDArray[np.bool_] = dist_sq <= radius * radius

        return self._subgraph_from_mask(within_mask)

    def extract_bbox(
        self,
        center: tuple[float, float] | npt.NDArray[np.floating] | None,
        width: float,
        height: float,
    ) -> MapGraph:
        """Extract a subgraph within a bounding box.

        Args:
            center: center of the bounding box as a tuple `(x, y)` or array. If `None`, the mean
                position of all nodes is used.
            width: full width of the bounding box.
            height: full height of the bounding box.

        Returns:
            A new `MapGraph` with only the nodes and edges inside the box.

        """
        if center is None:
            center_arr = np.mean(self.node_positions, axis=0)
        else:
            center_arr = np.asarray(center, dtype=np.float32)

        half_w = width / 2.0
        half_h = height / 2.0

        x = self.node_positions[:, 0]
        y = self.node_positions[:, 1]

        within_mask: npt.NDArray[np.bool_] = (
            (x >= center_arr[0] - half_w)
            & (x <= center_arr[0] + half_w)
            & (y >= center_arr[1] - half_h)
            & (y <= center_arr[1] + half_h)
        )

        return self._subgraph_from_mask(within_mask)

    def _subgraph_from_mask(
        self,
        node_mask: npt.NDArray[np.bool_],
    ) -> MapGraph:
        """Build a new `MapGraph` from a boolean node mask."""
        new_edge_indices, edge_mask = _extract_subgraph(node_mask, self.edge_indices)

        return MapGraph(
            node_positions=self.node_positions[node_mask],
            edge_indices=new_edge_indices,
            node_types=self.node_types[node_mask],
            edge_types=self.edge_types[edge_mask],
        )

    def __str__(self) -> str:
        """Return a string representation of the MapGraph."""
        return (
            f"MapGraph(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"node_positions_shape={self.node_positions.shape}, "
            f"edge_indices_shape={self.edge_indices.shape})"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation of the MapGraph."""
        return (
            f"MapGraph(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"node_positions={self.node_positions!r}, "
            f"edge_indices={self.edge_indices!r}, "
            f"node_types={self.node_types!r}, "
            f"edge_types={self.edge_types!r})"
        )


def _extract_subgraph(
    node_mask: npt.NDArray[np.bool_],
    edge_indices: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Extract a subgraph given a boolean node mask.

    This is a pure-NumPy replacement for `torch_geometric.utils.subgraph`. It filters edges so that
    both endpoints belong to the selected node subset and relabels node indices to be contiguous (0
    … K-1).

    Args:
        node_mask: boolean array of shape `(N,)` indicating which nodes to keep.
        edge_indices: integer array of shape `(2, M)` with source/destination
            indices for every edge.

    Returns:
        A tuple of:
        - `new_edge_indices`: relabelled edge index array of shape `(2, M')`.
        - `edge_mask`: boolean array of shape `(M,)` indicating which edges
          were kept.

    """
    src = edge_indices[0]
    dst = edge_indices[1]

    # Keep only edges where *both* endpoints are in the subset
    edge_mask: npt.NDArray[np.bool_] = node_mask[src] & node_mask[dst]

    # Build a mapping from old node index → new contiguous index.
    # Positions where node_mask is False get -1 (unused).
    remap = np.full(node_mask.shape[0], -1, dtype=np.int64)
    remap[node_mask] = np.arange(node_mask.sum(), dtype=np.int64)

    kept_src = remap[src[edge_mask]]
    kept_dst = remap[dst[edge_mask]]

    new_edge_indices = np.stack([kept_src, kept_dst], axis=0)
    return new_edge_indices, edge_mask
