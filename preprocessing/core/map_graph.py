from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
)

import numpy as np
import torch
from torch_geometric.utils import subgraph

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes


@dataclass(init=False, repr=False)
class MapGraph:
    """Represents a graph structure for a map, including nodes and edges."""

    def __init__(
        self,
        node_positions: torch.Tensor,
        edge_indices: torch.Tensor,
        node_types: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
        *,
        return_if_empty: bool = True,
    ) -> None:
        """Initialize a `MapGraph` instance.

        Args:
            node_positions: Tensor of shape (N, 2) representing the positions.
            edge_indices: Tensor of shape (2, M) representing the edges,
                where M is the number of edges.
            node_types: Tensor of shape (N,) representing the node types.
                Defaults to None, which means all nodes are of type 1.
            edge_types: Tensor of shape (M,) representing the edge types.
                Defaults to None, which means all edges are of type 1.
            return_if_empty: if True, allows empty graphs and returns empty
                tensors for node positions and edge indices. If False, raises
                a ValueError if the input tensors are empty.

        """
        # Check if the input tensors are empty
        if node_positions.numel() == 0 and edge_indices.numel() == 0:
            if return_if_empty:
                node_positions = torch.zeros((0, 2), dtype=torch.float)
                edge_indices = torch.zeros((2, 0), dtype=torch.long)
                node_types = torch.zeros((0,), dtype=torch.long)
                edge_types = torch.zeros((0,), dtype=torch.long)
            else:
                msg = (
                    "Node positions and edge indices cannot be empty."
                    " Set `return_if_empty=True` to allow empty graphs."
                )
                raise ValueError(msg)

        self.node_positions: torch.Tensor = node_positions
        self.edge_indices: torch.Tensor = edge_indices
        self.num_nodes: int = node_positions.shape[0]
        self.num_edges: int = edge_indices.shape[1] if edge_indices.numel() > 0 else 0
        self.node_types: torch.Tensor = (
            node_types
            if node_types is not None
            else torch.ones(node_positions.shape[0], dtype=torch.long)
        )

        self.edge_types: torch.Tensor = (
            edge_types if edge_types is not None else torch.ones(self.num_edges, dtype=torch.long)
        )

    def to_torch_graph(self) -> dict[Any, Any]:
        """Convert the MapGraph to a compatible format for PyTorch Geometric.

        Returns:
            dictionary with relevant data for PyTorch Geometric.

        """
        return {
            "map_point": {
                "num_nodes": self.num_nodes,
                "type": self.node_types,
                "position": self.node_positions,
            },
            ("map_point", "to", "map_point"): {
                "edge_index": self.edge_indices,
                "type": self.edge_types,
            },
        }

    @overload
    def extract_radius(
        self,
        center: torch.Tensor | tuple[float, float] | npt.NDArray[np.floating],
        radius: float,
        *,
        return_as_dict: Literal[False] = False,
    ) -> MapGraph: ...

    @overload
    def extract_radius(
        self,
        center: torch.Tensor | tuple[float, float] | npt.NDArray[np.floating],
        radius: float,
        *,
        return_as_dict: Literal[True],
    ) -> dict[Any, Any]: ...

    def extract_radius(
        self,
        center: torch.Tensor | tuple[float, float] | npt.NDArray[np.floating],
        radius: float,
        *,
        return_as_dict: bool = False,
    ) -> MapGraph | dict[Any, Any]:
        """Extract a subgraph within a specified radius from a center point.

        Args:
            center: center of the radius as a tensor or tuple of (x, y).
            radius: radius within which to extract nodes and edges.
            return_as_dict: if True, returns the graph in a format compatible with
                PyTorch Geometric. Otherwise, returns a `MapGraph` object.

        Returns:
            A `MapGraph` object or a dictionary with relevant data for PyTorch
            Geometric, depending on the value of `return_as_dict`.

        """
        if isinstance(center, tuple):
            # Convert tuple to tensor
            center = torch.tensor(center, dtype=torch.float32)

        elif isinstance(center, np.ndarray):
            # Convert numpy array to tensor
            center = torch.tensor(center, dtype=torch.float32)

        # Calculate distances from center to all nodes
        distances: torch.Tensor = torch.norm(self.node_positions - center, dim=-1)

        # Find nodes within radius
        within_radius_mask = distances <= radius

        # Use PyG's subgraph function to extract subgraph
        new_edge_indices, _, edge_mask = subgraph(
            subset=within_radius_mask,
            edge_index=self.edge_indices,
            relabel_nodes=True,
            return_edge_mask=True,
        )

        # Extract filtered data
        new_node_positions = self.node_positions[within_radius_mask]
        new_node_types = self.node_types[within_radius_mask]
        new_edge_types = self.edge_types[edge_mask]
        map_graph = MapGraph(
            node_positions=new_node_positions,
            edge_indices=new_edge_indices,
            node_types=new_node_types,
            edge_types=new_edge_types,
        )

        if return_as_dict:
            return map_graph.to_torch_graph()

        return map_graph

    @overload
    def extract_bbox(
        self,
        center: torch.Tensor | tuple[float, float] | npt.NDArray[np.floating],
        width: float,
        height: float,
        *,
        return_as_dict: Literal[False] = False,
    ) -> MapGraph: ...

    @overload
    def extract_bbox(
        self,
        center: torch.Tensor | tuple[float, float] | npt.NDArray[np.floating],
        width: float,
        height: float,
        *,
        return_as_dict: Literal[True],
    ) -> dict[Any, Any]: ...

    def extract_bbox(
        self,
        center: torch.Tensor | tuple[float, float] | npt.NDArray[np.floating],
        width: float,
        height: float,
        *,
        return_as_dict: bool = False,
    ) -> MapGraph | dict[Any, Any]:
        """Extract a subgraph within a bounding box.

        Args:
            center: center of the bounding box as a tensor or tuple of (x, y).
            width: width of the bounding box. If None, it is calculated as
                10% of the total width of the nodes. Defaults to None.
            height: height of the bounding box. If None, it is calculated as
                10% of the total height of the nodes. Defaults to None.
            return_as_dict: if True, returns the graph in a format compatible with
                PyTorch Geometric. Otherwise, returns a `MapGraph` object.

        Returns:
            A `MapGraph` object or a dictionary with relevant data for PyTorch
            Geometric, depending on the value of `return_as_dict`.

        """
        if center is None:
            # Mean position
            center = torch.mean(self.node_positions, dim=0)
        elif isinstance(center, tuple):
            # Convert tuple to tensor
            center = torch.tensor(center, dtype=torch.float32)

        if width is None:
            total_width = torch.max(self.node_positions[:, 0]) - torch.min(
                self.node_positions[:, 0],
            )
            width = total_width.item() * 0.1
        if height is None:
            total_height = torch.max(self.node_positions[:, 1]) - torch.min(
                self.node_positions[:, 1],
            )
            height = total_height.item() * 0.1

        # Calculate bounding box boundaries
        half_width = width / 2.0
        half_height = height / 2.0

        min_x = center[0] - half_width
        max_x = center[0] + half_width
        min_y = center[1] - half_height
        max_y = center[1] + half_height

        # Find nodes within bounding box
        x_coords = self.node_positions[:, 0]
        y_coords = self.node_positions[:, 1]

        within_bbox_mask = (
            (x_coords >= min_x) & (x_coords <= max_x) & (y_coords >= min_y) & (y_coords <= max_y)
        )

        # Use PyG's subgraph function to extract subgraph
        new_edge_indices, _, edge_mask = subgraph(
            subset=within_bbox_mask,
            edge_index=self.edge_indices,
            relabel_nodes=True,
            return_edge_mask=True,
        )

        # Extract filtered data
        new_node_positions = self.node_positions[within_bbox_mask]
        new_node_types = self.node_types[within_bbox_mask]
        new_edge_types = self.edge_types[edge_mask]

        map_graph = MapGraph(
            node_positions=new_node_positions,
            edge_indices=new_edge_indices,
            node_types=new_node_types,
            edge_types=new_edge_types,
        )

        if return_as_dict:
            return map_graph.to_torch_graph()

        return map_graph

    def __str__(self) -> str:
        """Return a string representation of the MapGraph."""
        return (
            f"MapGraph(num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"node_positions_shape={self.node_positions.shape}, "
            f"edge_indices_shape={self.edge_indices.shape})"
        )
