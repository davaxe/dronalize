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
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, auto
from math import ceil
from typing import Any, Generic, Literal, Protocol, TypedDict, TypeVar, overload

import numpy as np
import torch
from typing_extensions import Self

from preprocessing.core.categories import EdgeType
from preprocessing.core.map_graph import MapGraph

ID = TypeVar("ID", bound=Hashable)


@dataclass(slots=True, frozen=True)
class Implicit:
    """Indicates an implicit map context; map information is known without any additional info.

    Commonly this is the case when there is only one possible map.
    """

    tag: Literal["Implicit"] = "Implicit"


@dataclass(slots=True, frozen=True)
class Explicit:
    """Indicates an explicit map context; map information is provided explicitly for the scene."""

    identifier: str
    """Identifier for the map context, e.g., map name or token."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata required to determine the map."""
    tag: Literal["ImplicitWithInfo"] = "ImplicitWithInfo"


@dataclass(slots=True, frozen=True)
class Loaded:
    """Indicates a loaded map context; the map graph is directly provided."""

    map: MapGraph
    """The map graph directly provided for the scene."""
    tag: Literal["Loaded"] = "Loaded"


MapContext = Implicit | Explicit | Loaded | None


def implicit_map_context() -> Implicit:
    """Signify an implicit map context."""
    return Implicit()


def explicit_map_context(identifier: str, **metadata: Any) -> Explicit:  # noqa: ANN401
    """Create an explicit map context with the given identifier and metadata."""
    return Explicit(identifier=identifier, metadata=metadata or {})


def loaded_map_context(map_graph: MapGraph) -> Loaded:
    """Create a loaded map context with the given map graph."""
    return Loaded(map=map_graph)


def no_map_context() -> None:
    """Signify no map context."""


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

    def distance_sq_to(self, other: Self) -> float:
        """Calculate the squared Euclidean distance to another node."""
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2


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
    def with_id(cls, object_id: int, x: float, y: float, z: float = 0.0) -> IntIDNode:
        """Create an `IntIDNode` with a specified ID and coordinates."""
        node = cls(x, y, z)
        node.id = object_id
        return node

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

    edge_map: dict[EdgeType, EdgeType] = field(default_factory=dict, init=False)
    """Optionally remap edge types during graph building by adding entries"""

    _x: list[float] = field(default_factory=list, init=False)
    _y: list[float] = field(default_factory=list, init=False)
    _id_to_index: dict[ID, int] = field(default_factory=dict, init=False)

    _edge_src: list[int] = field(default_factory=list, init=False)
    _edge_dst: list[int] = field(default_factory=list, init=False)
    _edge_types: list[int] = field(default_factory=list, init=False)
    _seen_edges: set[tuple[int, int, int]] = field(default_factory=set, init=False)

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
        if node.id in self._id_to_index:
            return None if none_if_exists else node.id

        # 1. Assign next available index
        idx = len(self._x)
        self._id_to_index[node.id] = idx

        # 2. Store data in SoA (Structure of Arrays)
        self._x.append(node.x)
        self._y.append(node.y)

        # Note: We are discarding the 'node' object here to save RAM/Time.
        # If you strictly need access to 'self.nodes[id]' later in your code,
        # you would have to reconstruct it or keep the dict (at perf cost).

        return node.id

    def add_extra_node(self, node: NODE) -> ID:
        """Add an extra node to the graph.

        Args:
            node: node to add to the graph.

        Returns:
            The ID of the added node.

        """
        return self.add_node(node)

    def with_edge_map(self, edge_map: dict[EdgeType, EdgeType]) -> Self:
        """Set the edge mapping for remapping edge types during graph building.

        Args:
            edge_map: a dictionary mapping original `EdgeType` values to new
                `EdgeType` values. This allows for flexible remapping of edge
                types without modifying the underlying graph construction logic.

        Returns:
            The `GraphBuilder` instance with the updated edge mapping.

        """
        self.edge_map = edge_map
        return self

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

    def build_graph(self) -> MapGraph:
        """Build a graph representation of the map.

        This method converts the internal adjacency list and node positions into
        a `MapGraph` object.

        Returns:
            A `MapGraph` object representing the graph.

        """
        x_tensor = torch.tensor(self._x, dtype=torch.float32)
        y_tensor = torch.tensor(self._y, dtype=torch.float32)
        node_positions = torch.stack([x_tensor, y_tensor], dim=1)

        # 2. Convert Edges to Tensor
        edge_indices = torch.tensor([self._edge_src, self._edge_dst], dtype=torch.int64)
        edge_types = torch.tensor(self._edge_types, dtype=torch.int64)

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

        min_dist_sq = min_distance**2 if min_distance is not None else 0.0
        i, j = 0, 1
        while i < len(nodes) - 1:
            src, dst = nodes[i], nodes[j]
            distance_sq = src.distance_sq_to(dst)
            if distance_sq < min_dist_sq and j < len(nodes) - 1:
                j += 1
                continue
            if distance_sq < min_dist_sq:
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
        if from_id not in self._id_to_index or to_id not in self._id_to_index:
            # Optional: Raise error here if strict safety is needed
            return

        u = self._id_to_index[from_id]
        v = self._id_to_index[to_id]

        # Apply edge mapping immediately if present
        edge_type_val = self.edge_map.get(edge_type, edge_type)
        edge_type_int = int(edge_type_val)

        # Check for duplicates (much faster with a set of ints than checking lists)
        edge_key = (u, v, edge_type_int)
        if edge_key in self._seen_edges:
            return

        self._seen_edges.add(edge_key)

        # Append to COO lists
        self._edge_src.append(u)
        self._edge_dst.append(v)
        self._edge_types.append(edge_type_int)

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
        self._pending_paths.append({
            "nodes": nodes,
            "edge_type": edge_type,
            "is_polygon": is_polygon,
        })

    def _process_pending_paths(
        self,
        min_distance: float | None,
        interp_distance: float | None,
    ) -> None:
        for path_data in self._pending_paths:
            self.add_node_edges_loop_min_dist(
                nodes=path_data["nodes"],
                min_distance=min_distance,
                interp_distance=interp_distance,
                edge_type=path_data["edge_type"],
                is_polygon=path_data["is_polygon"],
            )

        # Clear buffer to prevent double-processing if built twice
        self._pending_paths.clear()

    @staticmethod
    def _resolve_distance(
        min_distance: float | None,
        interp_distance: float | None,
    ) -> tuple[float | None, float | None]:
        min_distance = min_distance if min_distance is not None else 0.0
        interp_distance = interp_distance if interp_distance is not None else np.inf
        if interp_distance <= 0.0:
            msg = "interp_distance must be greater than 0."
            raise ValueError(msg)
        if min_distance < 0.0 or min_distance > interp_distance:
            msg = (
                f"min_distance must be in the range [0, interp_distance] ([0, {interp_distance}])."
            )
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
    distance = math.hypot(dx, dy)
    if distance <= target_distance:
        yield InterpolationStage.LAST, src, dst
        return

    # Pre-calculate steps to avoid division in loop
    num_segments = ceil(distance / target_distance)
    step_x = dx / num_segments
    step_y = dy / num_segments

    prev_x, prev_y = src_x, src_y
    for i in range(1, num_segments + 1):
        curr_x = src_x + i * step_x
        curr_y = src_y + i * step_y

        stage = (
            InterpolationStage.FIRST
            if i == 1
            else InterpolationStage.LAST
            if i == num_segments
            else InterpolationStage.INTERMEDIATE
        )

        yield stage, (prev_x, prev_y), (curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y


ID = TypeVar("ID", bound=Hashable)


@dataclass(slots=True, frozen=True)
class Edges:
    """A collection of edges in a graph."""

    src_indices: list[int]
    dst_indices: list[int]
    edge_types: list[EdgeType]

    def to_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert the edges to PyTorch tensors.

        Returns:
            edge_indices (2xN), edge_types (Nx1) where N is the number of edges.

        """
        edge_index = torch.tensor([self.src_indices, self.dst_indices], dtype=torch.long)
        edge_attr = torch.tensor(self.edge_types, dtype=torch.long)
        return edge_index, edge_attr


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

    # Use flat lists instead of list of tuples
    src_indices: list[int] = []
    dst_indices: list[int] = []
    edge_types_list: list[EdgeType] = []  # Assuming EdgeType is IntEnum

    # Resolve edge_map.get outside the inner loop if possible,
    # or keep as is if dynamic.

    for from_id, to_list in adj_list.items():
        from_idx = id_to_index[from_id]
        for to_id, edge_type in to_list:
            src_indices.append(from_idx)
            dst_indices.append(id_to_index[to_id])

            # fast path for edge map
            etype = edge_map.get(edge_type, edge_type) if edge_map else edge_type
            edge_types_list.append(etype)

    # Return a structure that handles flat lists
    return Edges(src_indices, dst_indices, edge_types_list)
