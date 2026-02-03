# Copyright 2024-2025, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common data structures and utilities for map parsing and processing."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, auto
from math import ceil
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    Self,
    TypedDict,
    TypeVar,
    overload,
)

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from torch_geometric.utils import subgraph

from preprocessing.road_network.edge_type import EdgeType

if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes import Axes


ID = TypeVar("ID", bound=Hashable)

# CategoryStr is a type alias for a string literal that represents the category
# of a agent or object. This should maybe be moved to different file, e.g,
# preprocessing/common.py
CategoryStr = Literal[
    "car",
    "van",
    "trailer",
    "truck",
    "truck_bus",
    "tram",
    "bus",
    "motorcycle",
    "emergency_vehicle",
    "bicycle",
    "pedestrian",
    "tricycle",
    "animal",
    "static_object",
    "movable_object",
    "undefined",
]


@dataclass
class BaseMapObject(Protocol, Generic[ID]):
    """Base class for all map objects in a NuScenes map."""

    id: ID

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, Any]) -> Self:
        """Create an instance of the class from a dictionary."""
        ...

    @classmethod
    def try_from_dict(
        cls: type[Self],
        data: dict[str, Any],
    ) -> Self | None:
        """Try to create an instance from a dictionary, returning None on failure."""
        try:
            return cls.from_dict(data)
        except (ValueError, TypeError):
            return None


@dataclass(init=False)
class BaseNode(Protocol, Generic[ID]):
    """Protocol for a node in a map.

    This protocol defines the interface for a node in a map, which includes
    methods for calculating distances to other nodes.
    """

    id: ID
    x: float
    y: float
    z: float = 0.0

    def distance_to(self, other: Self) -> float:
        """Calculate the Euclidean distance to another node."""
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2,
        )


@dataclass(init=False)
class IntIdBaseMapObject(BaseMapObject[int]):
    """Base class for map objects with an integer ID.

    This class is used as a base for map objects that have an integer ID.
    It provides a common interface for creating instances from dictionaries.
    """

    def __init__(self, object_id: int | None = None) -> None:
        """Initialize an `IntIdBaseMapObject` with a unique integer ID."""
        cls = type(self)
        if not hasattr(cls, "_id_counter"):
            cls._id_counter = 0  # Initialize counter for the subclass
        if object_id is None:
            self.id = cls._id_counter
            cls._id_counter += 1
        else:
            self.id = object_id

    @classmethod
    def reset_counter(cls) -> None:
        """Reset the ID counter to 0."""
        cls._id_counter = 0


@dataclass(init=False)
class IntIDNode(IntIdBaseMapObject, BaseNode[int]):
    """A node in a 3D space with integer ID.

    The id is automatically assigned and unique for each instance, if not
    misused.
    """

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        """Initialize an `IntNode` with x, y, and optional z coordinates."""
        super().__init__()
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntIDNode:
        """Create an `IntNode` instance from a dictionary."""
        return IntIDNode(
            x=data["x"],
            y=data["y"],
            z=data.get("z", 0.0),
        )


class BaseEnum(IntEnum):
    """Base class for enums that can be created from strings.

    Note: the naming of the enum members should match the expected string
    values, e.g. if the string representation is "EXAMPLE", the enum member
    should be named `EXAMPLE`. This is to ensure that the `from_str` and
    `try_from_str` methods work correctly.
    """

    @classmethod
    def from_str(cls: type[Self], value: str) -> Self:
        """Convert a string to an enum member, raising `ValueError` if not found."""
        segment = cls.try_from_str(value)

        if segment is not None:
            return segment

        msg: str = (
            f"Enum value '{value}' is not recognized. "
            f"Available values: {', '.join(cls.__members__.keys())}."
        )
        raise ValueError(msg)

    @classmethod
    def try_from_str(cls: type[Self], value: str | None) -> Self | None:
        """Try to convert a string to an enum member, returning `None`  otherwise."""
        if value is None:
            return None
        return cls.__members__.get(value, None)

    @classmethod
    def from_int(cls: type[Self], value: int) -> Self:
        """Convert an integer to an enum member."""
        segment = cls.try_from_int(value)

        if segment is not None:
            return segment

        msg: str = (
            f"Enum value '{value}' is not recognized. "
            f"Available values: "
            f"{', '.join(map(str, cls._value2member_map_.keys()))}."
        )
        raise ValueError(msg)

    @classmethod
    def try_from_int(cls: type[Self], value: int | None) -> Self | None:
        """Try to convert a value to an enum member, returning `None` otherwise."""
        if value is None:
            return None
        return cls(value) if value in cls._value2member_map_ else None


NODE = TypeVar("NODE", bound=BaseNode)


class _PendingPathDict(TypedDict, Generic[NODE]):
    nodes: Sequence[NODE]
    edge_type: EdgeType | Sequence[EdgeType]
    is_polygon: bool


@dataclass
class GraphBuilder(ABC, Generic[ID, NODE]):
    """Abstract base class for graph builders.

    This class defines the interface for building graphs from map objects.
    Subclasses should implement the `build_graph` method to create a graph
    representation of the map.
    """

    nodes: dict[ID, NODE] = field(default_factory=dict, init=False)
    """All nodes in the graph."""

    id_adj_list: dict[ID, list[tuple[ID, EdgeType]]] = field(
        default_factory=dict,
        init=False,
    )
    """Adjacency list mapping node IDs to list of (neighbor_id, edge_type)."""

    edge_map: dict[EdgeType, EdgeType] = field(default_factory=dict, init=False)
    """Optionally remap edge types during graph building by adding entries"""

    extra_nodes: dict[ID, NODE] = field(default_factory=dict, init=False)
    """Extra nodes are nodes that are not connected to any other nodes, e.g., traffic lights."""

    _pending_paths: list[_PendingPathDict[NODE]] = field(
        default_factory=list,
        repr=False,
        init=False,
    )
    """List of pending paths to be added after all nodes are parsed. Used for
    lazy evaluation."""

    @abstractmethod
    def new_node(
        self,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> NODE:
        """Create a new node with the given coordinates."""
        ...

    @abstractmethod
    def build_impl(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        """Prepare internal structure for building the map graph.

        This means that the method should populate the internal data structures:
            - `self.nodes`
            - `self.id_adj_list`
            - `self.extra_nodes` (optional)
            - `self._pending_paths` (optional, if lazy evaluation is used)

        The function will be called automatically by the `build` method (if it
        is not overridden).

        Args:
            min_distance: minimum distance between nodes when adding edges.
                If None, no minimum distance is enforced.
            interp_distance: the target distance for interpolation. If None,
                no interpolation is performed.

        """

    @overload
    def add_node(
        self,
        node: NODE,
        *,
        none_if_exists: Literal[True],
    ) -> ID | None: ...

    @overload
    def add_node(
        self,
        node: NODE,
        *,
        none_if_exists: Literal[False] = False,
    ) -> ID: ...

    def add_node(self, node: NODE, *, none_if_exists: bool = False) -> ID | None:
        """Add a node to the graph.

        If the node already exists, it is not added again.

        Args:
            node: node to add to the graph.
            none_if_exists: if True, return None if the node already exists.

        Returns:
            The ID of the added node or `None` if the node already exists
            and `none_if_exists` is True.

        """
        if node.id not in self.nodes:
            self.nodes[node.id] = node
            self.id_adj_list[node.id] = []
            return node.id
        return node.id if not none_if_exists else None

    def add_extra_node(self, node: NODE) -> ID:
        """Add an extra node to the graph.

        Args:
            node: node to add to the graph.

        Returns:
            The ID of the added node.

        """
        if node.id not in self.extra_nodes:
            self.extra_nodes[node.id] = node
        return node.id

    def build(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> MapGraph:
        """Build a graph representation of the map.

        Args:
            min_distance: the minimum distance between nodes when adding edges.
                If None, no minimum distance is enforced.
            interp_distance: the target distance for interpolation. If None,
                no interpolation is performed.

        Returns:
            A `MapGraph` object representing the graph.

        """
        min_distance, interp_distance = self._resolve_distance(
            min_distance,
            interp_distance,
        )

        self.build_impl(
            min_distance=min_distance,
            interp_distance=interp_distance,
        )

        if self._pending_paths:
            self._process_pending_paths(
                min_distance=min_distance,
                interp_distance=interp_distance,
            )

        return self.build_graph()

    def build_graph(
        self,
        *,
        include_extra_nodes: bool = False,
    ) -> MapGraph:
        """Build a graph representation of the map.

        This method converts the internal adjacency list and node positions into
        a `MapGraph` object.

        Args:
            include_extra_nodes: whether to include extra nodes added by using
            `add_extra_node` method directly (e.g., traffic lights that are not
            part of any edges).

        Returns:
            A `MapGraph` object representing the graph.

        """
        node_positions = torch.zeros((len(self.nodes), 2), dtype=torch.float32)
        id_to_index: dict[ID, int] = {}
        for i, node in enumerate(self.nodes.values()):
            id_to_index[node.id] = i
            node_positions[i, 0] = node.x
            node_positions[i, 1] = node.y

        edge_indices, edge_types = get_edges_from_adj_list(
            self.id_adj_list,
            id_to_index,
            edge_map=self.edge_map,
        ).to_torch()

        if include_extra_nodes:
            n_nodes: int = len(self.extra_nodes)
            extra_node_positions = torch.zeros((n_nodes, 2), dtype=torch.float32)
            for i, node in enumerate(self.extra_nodes.values()):
                extra_node_positions[i, 0] = node.x
                extra_node_positions[i, 1] = node.y
            node_positions = torch.cat([node_positions, extra_node_positions], dim=0)

        return MapGraph(
            edge_indices=edge_indices,
            node_positions=node_positions,
            edge_types=edge_types,
        )

    def interpolate_edge(
        self,
        src: NODE,
        dst: NODE,
        interp_distance: float | None,
        edge_type: EdgeType,
    ) -> Iterable[tuple[ID, ID, EdgeType]]:
        """Interpolate an edge between two nodes.

        This method generates intermediate edges between two nodes, ensuring
        that the distance between consecutive points is at most `interp_distance`.

        Args:
            src: source node.
            dst: destination node.
            interp_distance: the target distance for interpolation. If None,
                no interpolation is performed.
            edge_type: type of the edge to be created.

        Yields:
            Tuples of (from_id, to_id, edge_type) for each interpolated edge.

        """
        # Get the actual node objects
        prev_id = self.add_node(src)
        for stage, _, (dst_x, dst_y) in interpolate_position(
            (src.x, src.y),
            (dst.x, dst.y),
            target_distance=interp_distance,
        ):
            if stage == InterpolationStage.LAST:
                new_id = self.add_node(dst)
            else:
                node = self.new_node(x=dst_x, y=dst_y)
                new_id = self.add_node(node)

            yield prev_id, new_id, edge_type
            prev_id = new_id

    def add_node_edges_loop(
        self,
        nodes: Sequence[NODE],
        *,
        interp_distance: float | None,
        edge_type: EdgeType | Sequence[EdgeType],
        is_polygon: bool = False,
    ) -> None:
        """Add edges between consecutive nodes in a given list.

        Args:
            nodes: a sequence of nodes to connect with edges.
            interp_distance: If None, no interpolation is performed.
            edge_type: type of the edge to be created. Either a single EdgeType or
                a sequence of EdgeTypes matching the number of edges to be created.
            is_polygon: whether the nodes form a polygon. If True, the last node
                is connected back to the first node to close the polygon.

        """
        if isinstance(edge_type, EdgeType):
            edge_type = [edge_type] * (len(nodes) - 1 + int(is_polygon))

        if len(edge_type) != len(nodes) - 1 + int(is_polygon):
            msg = (
                "Length of edge_type must be equal to the number of edges to be"
                f" created. Got {len(edge_type)} edge types for"
                f" {len(nodes) - 1 + int(is_polygon)} edges."
            )
            raise ValueError(msg)

        for i in range(len(nodes) - 1):
            src = nodes[i]
            dst = nodes[i + 1]
            self.add_node(src)
            self.add_node(dst)
            self.add_edges_from_iterable(
                self.interpolate_edge(
                    src,
                    dst,
                    interp_distance=interp_distance,
                    edge_type=edge_type[i],
                ),
            )

        if is_polygon:
            self.add_edges_from_iterable(
                self.interpolate_edge(
                    nodes[-1],
                    nodes[0],
                    interp_distance=interp_distance,
                    edge_type=edge_type[-1],
                ),
            )

    def add_node_edges_loop_min_dist(
        self,
        nodes: Sequence[NODE],
        *,
        min_distance: float | None,
        interp_distance: float | None,
        edge_type: EdgeType | Sequence[EdgeType],
        is_polygon: bool = False,
    ) -> None:
        """Add edges between consecutive nodes in a given list.

        The edges are only added if the distance between nodes is greater than
        the specified `min_distance` (greater than) threshold. This is useful
        for filtering out very short segments that may not be relevant for the
        graph representation.

        By using `min_distance` and `interp_distance` together, you can control
        both the minimum distance for adding edges and the maximum distance
        between interpolated points along those edges.

        Args:
            nodes: a sequence of nodes to connect with edges.
            min_distance: minimum distance threshold for adding edges.
            interp_distance: If None, no interpolation is performed.
            edge_type: type of the edge to be created. Either a single EdgeType
                or a sequence of EdgeTypes matching the number of edges to be
                created.
            is_polygon: whether the nodes form a polygon. Defaults to False.

        """
        if min_distance is None:
            self.add_node_edges_loop(
                nodes=nodes,
                interp_distance=interp_distance,
                edge_type=edge_type,
                is_polygon=is_polygon,
            )
            return

        if isinstance(edge_type, EdgeType):
            edge_type = [edge_type] * (len(nodes) - 1 + int(is_polygon))

        if len(edge_type) != len(nodes) - 1 + int(is_polygon):
            msg = (
                "Length of edge_type must be equal to the number of edges to be"
                f" created. Got {len(edge_type)} edge types for"
                f" {len(nodes) - 1 + int(is_polygon)} edges."
            )
            raise ValueError(msg)

        if interp_distance is not None and interp_distance < min_distance:
            msg = (
                "interp_distance must be greater than or equal to min_distance."
                f" Got interp_distance={interp_distance}, min_distance={min_distance}."
            )
            raise ValueError(msg)

        def add(src: NODE, dst: NODE, edge_type: EdgeType) -> None:
            self.add_node(dst)
            self.add_edges_from_iterable(
                self.interpolate_edge(
                    src,
                    dst,
                    interp_distance=interp_distance,
                    edge_type=edge_type,
                ),
            )

        i, j = 0, 1
        while i < len(nodes) - 1:
            src, dst = nodes[i], nodes[j]
            distance = src.distance_to(dst)
            if distance < min_distance and j < len(nodes) - 1:
                # Increment j to find a node that is >= min_distance away
                j += 1
                continue

            if distance < min_distance:
                # Always add last node, even if distance < min_distance
                add(src, dst, edge_type[i])
                break

            add(src, dst, edge_type[i])
            i = j
            j = i + 1

        if is_polygon:
            # Last edge is added in all cases if `is_polygon` is True
            self.add_edges_from_iterable(
                self.interpolate_edge(
                    nodes[-1],
                    nodes[0],
                    interp_distance=interp_distance,
                    edge_type=edge_type[-1],
                ),
            )

    def add_edges_from_iterable(
        self,
        edges: Iterable[tuple[ID, ID, EdgeType]],
    ) -> None:
        """Add edges to the graph from an iterable of edges.

        Args:
            edges: an iterable of tuples, where each tuple contains
                (from_id, to_id, edge_type).

        """
        for from_id, to_id, edge_type in edges:
            self.add_edge(from_id, to_id, edge_type)

    def add_edge(
        self,
        from_id: ID,
        to_id: ID,
        edge_type: EdgeType,
    ) -> None:
        """Add an edge to the graph.

        If the edge already exists, it is not added again.

        Args:
            from_id: ID of the source node.
            to_id: ID of the destination node.
            edge_type: type of the edge.

        """
        if from_id not in self.id_adj_list:
            self.id_adj_list[from_id] = []
        if (to_id, edge_type) not in self.id_adj_list[from_id]:
            self.id_adj_list[from_id].append((to_id, edge_type))

    def add_path_lazy(
        self,
        nodes: Sequence[NODE],
        edge_type: EdgeType | Sequence[EdgeType],
        *,
        is_polygon: bool = False,
    ) -> None:
        """Add a sequence (path) of nodes to be processed later.

        Args:
            nodes: nodes
            edge_type: edges between the nodes.
            is_polygon: _whether the nodes form a polygon. If True, the last node
                is connected back to the first node to close the polygon.

        """
        self._pending_paths.append(
            {
                "nodes": nodes,
                "edge_type": edge_type,
                "is_polygon": is_polygon,
            }
        )

    def _process_pending_paths(
        self,
        min_distance: float | None,
        interp_distance: float | None,
    ) -> None:
        for path_data in self._pending_paths:
            if min_distance is None:
                self.add_node_edges_loop(
                    nodes=path_data["nodes"],
                    interp_distance=interp_distance,
                    edge_type=path_data["edge_type"],
                    is_polygon=path_data["is_polygon"],
                )
            else:
                self.add_node_edges_loop_min_dist(
                    nodes=path_data["nodes"],
                    min_distance=min_distance,
                    interp_distance=interp_distance,
                    edge_type=path_data["edge_type"],
                    is_polygon=path_data["is_polygon"],
                )

        # Clear buffer to prevent double-processing if built twice
        self._pending_paths.clear()

    def _resolve_distance(
        self,
        min_distance: float | None,
        interp_distance: float | None,
    ) -> tuple[float | None, float | None]:
        _min_distance = min_distance if min_distance is not None else 0.0
        _interp_distance = interp_distance if interp_distance is not None else np.inf
        if _interp_distance <= 0.0:
            msg = "interp_distance must be greater than 0."
            raise ValueError(msg)
        if _min_distance < 0.0 or _min_distance > _interp_distance:
            msg = f"min_distance must be in the range [0, interp_distance] ([0, {_interp_distance}])."
            raise ValueError(msg)

        return min_distance, interp_distance


class InterpolationStage(IntEnum):
    """Stages of interpolation for a path between two points.

    This enum is used in `interpolate_position` to indicate the stage of the
    interpolation process. It helps to distinguish between the first,
    intermediate, and last segments of the path.
    """

    FIRST = auto()
    INTERMEDIATE = auto()
    LAST = auto()


def interpolate_position(
    src: tuple[float, float],
    dst: tuple[float, float],
    *,
    target_distance: float | None = 1.0,
) -> Iterable[tuple[InterpolationStage, tuple[float, float], tuple[float, float]]]:
    """Interpolate positions between two points.

    This function generates intermediate points between two source and
    destination coordinates, ensuring that the distance between consecutive
    points is at most `target_distance`.

    Args:
        src: source coordinates as a tuple of (x, y).
        dst: destination coordinates as a tuple of (x, y).
        target_distance: the maximum distance between consecutive points.
            Default is 1.0.

    Yields:
        A tuple (`InterpolationStage`, `src`, `dst`) for each segment of the
            interpolated path.

    """
    if target_distance is None or target_distance <= 0:
        yield InterpolationStage.LAST, src, dst
        return

    src_x, src_y = src
    dst_x, dst_y = dst
    dx: float = dst_x - src_x
    dy: float = dst_y - src_y
    distance: float = (dx**2 + dy**2) ** 0.5
    if distance <= target_distance:
        yield InterpolationStage.LAST, src, dst
        return

    num_segments: int = ceil(distance / target_distance)

    prev_x: float = src_x
    prev_y: float = src_y
    for i in range(1, num_segments + 1):
        t = i / num_segments
        interp_x = src_x + t * dx
        interp_y = src_y + t * dy
        stage = (
            InterpolationStage.FIRST
            if i == 0
            else InterpolationStage.LAST
            if i == num_segments - 1
            else InterpolationStage.INTERMEDIATE
        )

        yield stage, (prev_x, prev_y), (interp_x, interp_y)

        prev_x = interp_x
        prev_y = interp_y


ID = TypeVar("ID", bound=Hashable)


@dataclass
class Edges:
    """A collection of edges in a graph."""

    edge_indices: list[tuple[int, int]]
    edge_types: list[EdgeType]

    def to_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert the edges to PyTorch tensors.

        Returns:
            edge_indices (2xN), edge_types (Nx1) where N is the number of edges.

        """
        return (
            torch.tensor(self.edge_indices, dtype=torch.int64).t(),
            torch.tensor(self.edge_types, dtype=torch.int64),
        )


def get_edges_from_adj_list(
    adj_list: dict[ID, list[tuple[ID, EdgeType]]],
    id_to_index: dict[ID, int] | None = None,
    edge_map: dict[EdgeType, EdgeType] | None = None,
) -> Edges:
    """Get edges from an adjacency list.

    Args:
        adj_list: adjacency list where keys are node IDs and values are lists of
            tuples containing destination node IDs and edge types.
        id_to_index: a mapping from node IDs to their indices in the graph.
        edge_map: a optional mapping from edge type if needed to remap edge types.

    Returns:
        An `Edges` object containing the edge indices and types.

    """
    if id_to_index is None:
        id_to_index = {node_id: index for index, node_id in enumerate(adj_list.keys())}

    if edge_map is None:
        edge_map = {}

    edge_indices: list[tuple[int, int]] = []
    edge_types: list[EdgeType] = []
    for from_id, to_list in adj_list.items():
        for to_id, edge_type in to_list:
            from_index = id_to_index[from_id]
            to_index = id_to_index[to_id]
            edge_indices.append((from_index, to_index))
            edge_types.append(edge_map.get(edge_type, edge_type))

    return Edges(edge_indices, edge_types)


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
            edge_types
            if edge_types is not None
            else torch.ones(self.num_edges, dtype=torch.long)
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

    def plot_graph(self) -> Axes:
        """Visualize the map graph using Matplotlib."""
        _fig, ax = plt.subplots(figsize=(10, 10))

        # Dictionary to group line segments by their EdgeType
        # Key: EdgeType, Value: list of segments [(x1, y1), (x2, y2)]
        lines_by_type: defaultdict[EdgeType, list[list[tuple[float, float]]]] = (
            defaultdict(list)
        )

        # 1. Group segments by type
        for i in range(self.edge_indices.shape[1]):
            src_idx = int(self.edge_indices[0, i].item())
            dst_idx = int(self.edge_indices[1, i].item())
            src_pos = self.node_positions[src_idx].numpy()
            dst_pos = self.node_positions[dst_idx].numpy()
            edge_type = EdgeType(self.edge_types[i].item())
            lines_by_type[edge_type].append([tuple(src_pos), tuple(dst_pos)])

        legend_elements = []

        for edge_type, segments in lines_by_type.items():
            style_kwargs = edge_type.edge_style()
            _ = style_kwargs.pop("dashes", None)

            lc = LineCollection(segments, **style_kwargs)
            ax.add_collection(lc)
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    label=f"Edge Type {edge_type.name}",
                    **style_kwargs,
                ),
            )

        ax.autoscale()
        ax.set_aspect("equal", "box")
        ax.set_title("Argoverse1 Map Graph")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.legend(
            handles=legend_elements,
            title="Edge Types",
            loc="upper right",
        )

        return ax

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
        distances = torch.norm(self.node_positions - center, dim=-1)

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
            (x_coords >= min_x)
            & (x_coords <= max_x)
            & (y_coords >= min_y)
            & (y_coords <= max_y)
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
