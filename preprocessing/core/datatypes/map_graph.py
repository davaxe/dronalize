from __future__ import annotations

import multiprocessing.shared_memory as shm
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Self

from preprocessing.core._compat import require_optional

if TYPE_CHECKING:
    from types import TracebackType

    import numpy.typing as npt


class SharedMapGraph:
    """Context manager for accessing a `MapGraph` stored in shared memory."""

    def __init__(self, shared_name: str) -> None:
        """Initialize the shared memory context manager."""
        self._shared_name = shared_name
        self._shared: shm.SharedMemory | None = None
        self._map_graph: MapGraph | None = None

    def open(self) -> MapGraph:
        """Open the shared memory and return a `MapGraph` instance.

        Returns
        -------
        MapGraph
            The `MapGraph` instance loaded from shared memory.

        Raises
        ------
        RuntimeError
            If the shared memory is already open or if there is an issue
            accessing it (e.g., it doesn't exist or has been released).

        """
        if self._shared is not None:
            msg = f"Shared memory with name '{self._shared_name}' is already open."
            raise RuntimeError(msg)

        self._shared = shm.SharedMemory(name=self._shared_name)

        if self._shared.buf is None:
            msg = f"Failed to access shared memory buffer with name '{self._shared_name}'"
            raise RuntimeError(msg)

        # Store the instance so we can release its pointers later
        self._map_graph = MapGraph.from_buffer(self._shared.buf, read_only=True)
        return self._map_graph

    def close(self) -> None:
        """Close the shared memory and release resources.

        Raises
        ------
        RuntimeError
            If the shared memory is not currently open.

        """
        if self._shared is None:
            msg = f"Shared memory with name '{self._shared_name}' is not open."
            raise RuntimeError(msg)

        # Release the MapGraph's references to the shared memory arrays to allow cleanup.
        if self._map_graph is not None:
            self._map_graph.release()
            self._map_graph = None

        self._shared.close()

    def __enter__(self) -> MapGraph:
        """Access the `MapGraph` from shared memory.

        Returns
        -------
        MapGraph
            The `MapGraph` instance loaded from shared memory.

        """
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_t: TracebackType | None,
    ) -> None:
        """Release resources when exiting the context.

        Parameters
        ----------
        exc_type : type[BaseException] or None
            The exception type, if any.
        exc_val : BaseException or None
            The exception value, if any.
        exc_t : TracebackType or None
            The traceback, if any.

        """
        self.close()

    def __del__(self) -> None:
        """Ensure shared memory is closed."""
        # Ensure shared memory is closed if the context manager was not used properly.
        if self._shared is not None:
            self._shared.close()

        if self._map_graph is not None:
            self._map_graph.release()


@dataclass(init=False, repr=False, slots=True)
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

        Parameters
        ----------
        node_positions : ndarray of float32, shape (N, 2)
            Array of node positions.
        edge_indices : ndarray of int64, shape (2, M)
            Array of edge endpoint indices.
        node_types : ndarray of int64, shape (N,), optional
            Per-node type labels. If None, all nodes are assigned type 1.
        edge_types : ndarray of int64, shape (M,), optional
            Per-edge type labels. If None, all edges are assigned type 1.
        return_if_empty : bool, optional
            If True, allows empty graphs and returns empty arrays. If False,
            raises `ValueError` when both `node_positions` and
            `edge_indices` are empty.

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

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return int(self.node_positions.shape[0])

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        return int(self.edge_indices.shape[1])

    def to_shared(self) -> shm.SharedMemory:
        """Serialize the `MapGraph` to a shared memory block, including dimensions.

        Returns
        -------
        SharedMemory
            `SharedMemory` object containing the serialized graph data.

        """
        # Pack the dimensions into an int64 array to serve as the header
        header = np.array([self.num_nodes, self.num_edges], dtype=np.int64)

        arrays_to_serialize = (
            header,
            self.node_positions,
            self.edge_indices,
            self.node_types,
            self.edge_types,
        )

        total_bytes = sum(arr.nbytes for arr in arrays_to_serialize)
        shared = shm.SharedMemory(create=True, size=total_bytes)

        offset = 0
        for arr in arrays_to_serialize:
            target_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shared.buf, offset=offset)
            np.copyto(target_arr, arr)
            offset += arr.nbytes

        return shared

    @classmethod
    def from_shared(cls, shared_name: str) -> SharedMapGraph:
        """Create a context manager for accessing a graph in shared memory.

        Parameters
        ----------
        shared_name : str
            Name of the shared memory block containing the graph data.

        """
        return SharedMapGraph(shared_name)

    @classmethod
    def from_buffer(cls, buf: memoryview, *, read_only: bool = True) -> Self:
        """Deserialize a `MapGraph` from a bytes buffer with embedded metadata."""
        offset = 0

        # Read the first two int64 values to get the dimensions
        header = np.frombuffer(buf, dtype=np.int64, count=2, offset=offset)
        n, m = int(header[0]), int(header[1])
        offset += header.nbytes

        positions = np.frombuffer(buf, dtype=np.float32, count=n * 2, offset=offset).reshape((n, 2))
        offset += positions.nbytes

        indices = np.frombuffer(buf, dtype=np.int64, count=m * 2, offset=offset).reshape((2, m))
        offset += indices.nbytes

        node_types = np.frombuffer(buf, dtype=np.int64, count=n, offset=offset)
        offset += node_types.nbytes

        edge_types = np.frombuffer(buf, dtype=np.int64, count=m, offset=offset)

        if read_only:
            for arr in (positions, indices, node_types, edge_types):
                arr.flags.writeable = False

        return cls(
            node_positions=positions,
            edge_indices=indices,
            node_types=node_types,
            edge_types=edge_types,
        )

    def release(self) -> None:
        """Release references to the internal arrays to allow shared memory cleanup.

        This will clear the internal arrays and is not intended for general use.
        """
        self.node_positions = np.array([], dtype=np.float32).reshape(0, 2)
        self.edge_indices = np.array([], dtype=np.int64).reshape(2, 0)
        self.node_types = np.array([], dtype=np.int64)
        self.edge_types = np.array([], dtype=np.int64)

    def to_torch_graph(self) -> dict[Any, Any]:
        """Convert the `MapGraph` to a format compatible with PyTorch Geometric.

        Uses `torch.from_numpy` for zero-copy conversion (the returned tensors
        share the same underlying memory as the NumPy arrays).

        Returns
        -------
        dict
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

    def extract_radius(
        self,
        center: tuple[float, float] | npt.NDArray[np.floating],
        radius: float,
    ) -> MapGraph:
        """Extract a subgraph within a specified radius from a center point.

        Parameters
        ----------
        center : tuple[float, float] or ndarray
            Center of the radius as a tuple `(x, y)` or array.
        radius : float
            Radius within which to extract nodes and edges.

        Returns
        -------
        MapGraph
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

        Parameters
        ----------
        center : tuple[float, float] or ndarray or None
            Center of the bounding box as a tuple `(x, y)` or array. If
            `None`, the mean position of all nodes is used.
        width : float
            Full width of the bounding box.
        height : float
            Full height of the bounding box.

        Returns
        -------
        MapGraph
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

    This is a pure-NumPy replacement for `torch_geometric.utils.subgraph`. It
    filters edges so that both endpoints belong to the selected node subset and
    relabels node indices to be contiguous (0 … K-1).

    Parameters
    ----------
    node_mask : ndarray of bool, shape (N,)
        Boolean array indicating which nodes to keep.
    edge_indices : ndarray of int64, shape (2, M)
        Source/destination indices for every edge.

    Returns
    -------
    new_edge_indices : ndarray of int64, shape (2, M')
        Relabelled edge index array for the kept edges.
    edge_mask : ndarray of bool, shape (M,)
        Boolean array indicating which edges were kept.

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
