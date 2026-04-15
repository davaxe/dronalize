"""Generic map-graph builder abstractions and geometry helpers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from itertools import starmap
from math import ceil
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, TypedDict, override

from dronalize.core.categories import EdgeType
from dronalize.core.maps import MapGraph

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

Point = tuple[float, float]
"""A 2-D point as `(x, y)`."""


class _PendingPathDict(TypedDict):
    points: Sequence[Point]
    edge_type: EdgeType | Sequence[EdgeType]
    is_polygon: bool


class MapBuilder(Protocol):
    """Minimal protocol for building a map graph."""

    def build(
        self, min_distance: float | None = None, interp_distance: float | None = None
    ) -> MapGraph:
        """Build the final `MapGraph`.

        Parameters
        ----------
        min_distance : float, optional
            Minimum distance between nodes when adding edges.
        interp_distance : float, optional
            Target distance for interpolation.

        Returns
        -------
        MapGraph
            The constructed graph.

        """
        ...


@dataclass
class BaseMapBuilder(MapBuilder, ABC):
    """Abstract base class for map builders.

    Subclasses only need to implement `build_impl` which populates
    the internal graph structure using the helper methods provided here
    (`add_node`, `add_edge`, `add_path`, `add_path_lazy`, etc.).

    Nodes are identified by **integer IDs** that are assigned automatically
    when calling `add_node`.  There is no need for an external node
    class — the builder stores coordinates directly in an efficient
    Structure-of-Arrays layout.
    """

    edge_map: dict[EdgeType, EdgeType] = field(default_factory=dict, init=False)
    """Optionally remap edge types during graph building by adding entries."""

    _node_id_counter: int = field(default=0, init=False, repr=False)
    """Auto-incrementing counter for unique node IDs."""

    _x: list[float] = field(default_factory=list, init=False)
    _y: list[float] = field(default_factory=list, init=False)
    _id_to_index: dict[int, int] = field(default_factory=dict, init=False)

    _edge_src: list[int] = field(default_factory=list, init=False)
    _edge_dst: list[int] = field(default_factory=list, init=False)
    _edge_types: list[int] = field(default_factory=list, init=False)
    _seen_edges: set[tuple[int, int, int]] = field(default_factory=set, init=False)

    _pending_paths: list[_PendingPathDict] = field(default_factory=list, repr=False, init=False)
    """Paths queued via `add_path_lazy` for deferred processing."""

    def _next_node_id(self) -> int:
        """Return the next unique integer node ID and advance the counter.

        Returns
        -------
        int
            A unique, monotonically increasing node ID.

        """
        node_id = self._node_id_counter
        self._node_id_counter += 1
        return node_id

    def add_node(self, x: float, y: float) -> int:
        """Add a node at *(x, y)* and return its unique integer ID.

        If a node with the same ID already exists this is a no-op and the
        existing ID is returned (though in practice every call allocates a
        fresh ID, so duplicates only arise when `add_node_with_id` is
        used).

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.

        Returns
        -------
        int
            The ID assigned to the newly created node.

        """
        node_id = self._next_node_id()
        idx = len(self._x)
        self._id_to_index[node_id] = idx
        self._x.append(x)
        self._y.append(y)
        return node_id

    def add_node_with_id(self, node_id: int, x: float, y: float) -> int:
        """Add a node with a *specific* pre-assigned ID.

        If `node_id` has already been registered the call is a no-op and
        the existing ID is returned.

        Parameters
        ----------
        node_id : int
            The integer ID to assign.
        x : float
            X coordinate.
        y : float
            Y coordinate.

        Returns
        -------
        int
            `node_id` (echoed back for convenience).

        """
        if node_id in self._id_to_index:
            return node_id

        idx = len(self._x)
        self._id_to_index[node_id] = idx
        self._x.append(x)
        self._y.append(y)
        if node_id >= self._node_id_counter:
            self._node_id_counter = node_id + 1

        return node_id

    @abstractmethod
    def build_impl(
        self, min_distance: float | None = None, interp_distance: float | None = None
    ) -> None:
        """Prepare internal structure for building the map graph.

        Subclasses populate the graph by calling the helper methods provided
        by `MapBuilder` (`add_node`, `add_edge`, `add_path`,
        `add_path_lazy`, etc.).

        Parameters
        ----------
        min_distance : float, optional
            Minimum distance between nodes when adding edges.
            If None, no minimum distance is enforced.
        interp_distance : float, optional
            The target distance for interpolation. If None, no interpolation
            is performed.

        """

    def with_edge_map(self, edge_map: dict[EdgeType, EdgeType]) -> Self:
        """Set the edge mapping for remapping edge types during graph building.

        Parameters
        ----------
        edge_map : dict[EdgeType, EdgeType]
            A dictionary mapping original `EdgeType` values to new
            `EdgeType` values.

        Returns
        -------
        Self
            The `MapBuilder` instance with the updated edge mapping.

        """
        self.edge_map = edge_map
        return self

    def add_edge(self, from_id: int, to_id: int, edge_type: EdgeType) -> None:
        """Add a directed edge to the graph.

        If the edge already exists it is silently skipped.

        Parameters
        ----------
        from_id : int
            ID of the source node.
        to_id : int
            ID of the destination node.
        edge_type : EdgeType
            Type of the edge.

        """
        if from_id not in self._id_to_index or to_id not in self._id_to_index:
            return

        u = self._id_to_index[from_id]
        v = self._id_to_index[to_id]

        edge_type_val = self.edge_map.get(edge_type, edge_type)
        edge_type_int = int(edge_type_val)

        edge_key = (u, v, edge_type_int)
        if edge_key in self._seen_edges:
            return

        self._seen_edges.add(edge_key)

        self._edge_src.append(u)
        self._edge_dst.append(v)
        self._edge_types.append(edge_type_int)

    def add_edges_from_iterable(self, edges: Iterable[tuple[int, int, EdgeType]]) -> None:
        """Add edges to the graph from an iterable.

        Parameters
        ----------
        edges : Iterable[tuple[int, int, EdgeType]]
            An iterable of `(from_id, to_id, edge_type)` tuples.

        """
        for from_id, to_id, edge_type in edges:
            self.add_edge(from_id, to_id, edge_type)

    def interpolate_edge(
        self,
        src_id: int,
        src_xy: Point,
        dst_id: int,
        dst_xy: Point,
        interp_distance: float | None,
        edge_type: EdgeType,
    ) -> Iterable[tuple[int, int, EdgeType]]:
        """Interpolate between two *already-added* nodes.

        Intermediate nodes are created automatically so that consecutive
        points are at most `interp_distance` apart.

        Parameters
        ----------
        src_id : int
            ID of the source node (must already exist in the builder).
        src_xy : Point
            `(x, y)` of the source node.
        dst_id : int
            ID of the destination node (must already exist in the builder).
        dst_xy : Point
            `(x, y)` of the destination node.
        interp_distance : float or None
            Target max distance between consecutive points.  `None` means
            no interpolation — a single edge is emitted.
        edge_type : EdgeType
            Edge type for every generated edge.

        Yields
        ------
        tuple[int, int, EdgeType]
            `(from_id, to_id, edge_type)` for each (interpolated) edge.

        """
        prev_id = src_id
        for stage, _, (ix, iy) in interpolate_position(
            src_xy, dst_xy, target_distance=interp_distance
        ):
            new_id = dst_id if stage == InterpolationStage.LAST else self.add_node(ix, iy)
            yield prev_id, new_id, edge_type
            prev_id = new_id

    def add_path(
        self,
        points: Sequence[Point],
        edge_type: EdgeType | Sequence[EdgeType],
        *,
        is_polygon: bool = False,
        interp_distance: float | None = None,
    ) -> list[int]:
        """Add a sequence of points as connected nodes and return their IDs.

        Parameters
        ----------
        points : Sequence[Point]
            Sequence of `(x, y)` coordinates.
        edge_type : EdgeType or Sequence[EdgeType]
            A single edge type (applied to every edge) or a per-edge
            sequence whose length must equal the number of edges.
        is_polygon : bool, optional
            If `True` the last point is connected back to the first.
        interp_distance : float or None, optional
            If given, edges longer than this are interpolated.

        Returns
        -------
        list[int]
            The node IDs (one per input point, plus any interpolation
            nodes that were inserted).

        """
        if len(points) < 2:
            if points:
                return [self.add_node(*points[0])]
            return []

        n_edges = len(points) - 1 + int(is_polygon)
        if isinstance(edge_type, EdgeType):
            edge_types: list[EdgeType] = [edge_type] * n_edges
        else:
            edge_types = list(edge_type)
            if len(edge_types) != n_edges:
                msg = (
                    f"Length of edge_type must equal the number of edges. "
                    f"Got {len(edge_types)} for {n_edges} edges."
                )
                raise ValueError(msg)

        # Create nodes for all input points first.
        ids = list(starmap(self.add_node, points))

        for i in range(len(points) - 1):
            self.add_edges_from_iterable(
                self.interpolate_edge(
                    ids[i], points[i], ids[i + 1], points[i + 1], interp_distance, edge_types[i]
                )
            )

        if is_polygon:
            self.add_edges_from_iterable(
                self.interpolate_edge(
                    ids[-1], points[-1], ids[0], points[0], interp_distance, edge_types[-1]
                )
            )

        return ids

    def add_node_edges_loop(
        self,
        points: Sequence[Point],
        *,
        interp_distance: float | None,
        edge_type: EdgeType | Sequence[EdgeType],
        is_polygon: bool = False,
    ) -> None:
        """Add edges between consecutive points.

        Parameters
        ----------
        points : Sequence[Point]
            `(x, y)` coordinates to connect.
        interp_distance : float or None
            Interpolation target distance.
        edge_type : EdgeType or Sequence[EdgeType]
            Edge type(s).
        is_polygon : bool, optional
            Connect last point back to first.

        """
        _ = self.add_path(points, edge_type, is_polygon=is_polygon, interp_distance=interp_distance)

    def add_node_edges_loop_min_dist(
        self,
        points: Sequence[Point],
        *,
        min_distance: float | None,
        interp_distance: float | None,
        edge_type: EdgeType | Sequence[EdgeType],
        is_polygon: bool = False,
    ) -> None:
        """Add edges between consecutive points, skipping short segments.

        Parameters
        ----------
        points : Sequence[Point]
            `(x, y)` coordinates to connect.
        min_distance : float or None
            Skip edges shorter than this.  `None` disables filtering.
        interp_distance : float or None
            Interpolation target distance.
        edge_type : EdgeType or Sequence[EdgeType]
            Edge type(s).
        is_polygon : bool, optional
            Connect last point back to first.

        """
        if min_distance is None:
            _ = self.add_path(
                points, edge_type, is_polygon=is_polygon, interp_distance=interp_distance
            )
            return

        if len(points) < 2:
            if points:
                _ = self.add_node(*points[0])
            return

        n_edges = len(points) - 1 + int(is_polygon)
        edge_types = _normalize_edge_types(edge_type, n_edges)
        if interp_distance is not None and interp_distance < min_distance:
            msg = (
                "interp_distance must be >= min_distance. "
                f"Got interp_distance={interp_distance}, min_distance={min_distance}."
            )
            raise ValueError(msg)

        min_dist_sq = min_distance**2

        def _dist_sq(a: Point, b: Point) -> float:
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        # Add the very first node.
        prev_id = self.add_node(*points[0])
        prev_pt = points[0]

        i, j = 0, 1
        while i < len(points) - 1:
            dst_pt = points[j]
            dsq = _dist_sq(prev_pt, dst_pt)
            if dsq < min_dist_sq and j < len(points) - 1:
                j += 1
                continue

            dst_id = self.add_node(*dst_pt)
            self.add_edges_from_iterable(
                self.interpolate_edge(
                    prev_id, prev_pt, dst_id, dst_pt, interp_distance, edge_types[i]
                )
            )
            prev_id = dst_id
            prev_pt = dst_pt
            i = j
            j = i + 1

        if is_polygon:
            first_pt = points[0]
            # Close the polygon by connecting back to the first point.
            close_id = self.add_node(*first_pt)
            self.add_edges_from_iterable(
                self.interpolate_edge(
                    prev_id, prev_pt, close_id, first_pt, interp_distance, edge_types[-1]
                )
            )

    def add_path_lazy(
        self,
        points: Sequence[Point],
        edge_type: EdgeType | Sequence[EdgeType],
        *,
        is_polygon: bool = False,
    ) -> None:
        """Queue a path for deferred processing by `build`.

        Parameters
        ----------
        points : Sequence[Point]
            `(x, y)` coordinates forming the path.
        edge_type : EdgeType or Sequence[EdgeType]
            Edge type(s).
        is_polygon : bool, optional
            Whether the path forms a closed polygon.

        """
        self._pending_paths.append({
            "points": points,
            "edge_type": edge_type,
            "is_polygon": is_polygon,
        })

    def _process_pending_paths(
        self, min_distance: float | None, interp_distance: float | None
    ) -> None:
        for path_data in self._pending_paths:
            self.add_node_edges_loop_min_dist(
                points=path_data["points"],
                min_distance=min_distance,
                interp_distance=interp_distance,
                edge_type=path_data["edge_type"],
                is_polygon=path_data["is_polygon"],
            )
        self._pending_paths.clear()

    @override
    def build(
        self, min_distance: float | None = None, interp_distance: float | None = None
    ) -> MapGraph:
        """Build the final `MapGraph`.

        Parameters
        ----------
        min_distance : float, optional
            Minimum distance between nodes when adding edges.
        interp_distance : float, optional
            Target distance for interpolation.

        Returns
        -------
        MapGraph
            The constructed graph.

        """
        min_distance, interp_distance = self._resolve_distance(min_distance, interp_distance)

        self.build_impl(min_distance=min_distance, interp_distance=interp_distance)

        if self._pending_paths:
            self._process_pending_paths(min_distance=min_distance, interp_distance=interp_distance)

        return self._to_map_graph()

    def _to_map_graph(self) -> MapGraph:
        """Convert the internal state into a `MapGraph`.

        Returns
        -------
        MapGraph
            The graph built from the accumulated nodes and edges.

        """
        node_positions = np.column_stack([
            np.array(self._x, dtype=np.float64),
            np.array(self._y, dtype=np.float64),
        ])

        edge_indices = np.array([self._edge_src, self._edge_dst], dtype=np.int32)
        edge_types = np.array(self._edge_types, dtype=np.int32)

        return MapGraph(
            edge_indices=edge_indices, node_positions=node_positions, edge_types=edge_types
        )

    def node_xy(self, node_id: int) -> Point:
        """Return `(x, y)` for an already-added node."""
        idx = self._id_to_index[node_id]
        return (self._x[idx], self._y[idx])

    def node_distance(self, id_a: int, id_b: int) -> float:
        """Euclidean distance between two nodes."""
        idx_a = self._id_to_index[id_a]
        idx_b = self._id_to_index[id_b]
        dx = self._x[idx_a] - self._x[idx_b]
        dy = self._y[idx_a] - self._y[idx_b]
        return math.sqrt(dx * dx + dy * dy)

    def edge_exists(self, from_id: int, to_id: int, edge_type: EdgeType) -> bool:
        """Check whether a specific edge is already present."""
        if from_id not in self._id_to_index or to_id not in self._id_to_index:
            return True  # treat missing nodes as "exists" → skip adding
        u = self._id_to_index[from_id]
        v = self._id_to_index[to_id]
        return (u, v, int(edge_type)) in self._seen_edges

    @staticmethod
    def _resolve_distance(
        min_distance: float | None, interp_distance: float | None
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
    """Stage of an interpolation step."""

    INTERMEDIATE = auto()
    LAST = auto()


def interpolate_position(
    src: tuple[float, float], dst: tuple[float, float], target_distance: float | None = None
) -> Iterable[tuple[InterpolationStage, int, tuple[float, float]]]:
    """Interpolate positions between *src* and *dst*.

    Generates intermediate positions so that the distance between
    consecutive points does not exceed `target_distance`.

    Parameters
    ----------
    src : tuple[float, float]
        Source `(x, y)` position.
    dst : tuple[float, float]
        Destination `(x, y)` position.
    target_distance : float or None
        Maximum distance between consecutive points.  If `None` or
        infinite, a single step (the destination) is yielded.

    Yields
    ------
    tuple[InterpolationStage, int, tuple[float, float]]
        `(stage, step_index, (x, y))` for each interpolated position.

    """
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    distance = math.sqrt(dx * dx + dy * dy)

    if target_distance is None or target_distance == float("inf") or distance <= target_distance:
        yield InterpolationStage.LAST, 0, dst
        return

    num_steps = ceil(distance / target_distance)
    step_x = dx / num_steps
    step_y = dy / num_steps

    for i in range(1, num_steps):
        x = src[0] + step_x * i
        y = src[1] + step_y * i
        yield InterpolationStage.INTERMEDIATE, i - 1, (x, y)

    yield InterpolationStage.LAST, num_steps - 1, dst


@dataclass(frozen=True, slots=True)
class Edges:
    """Container for directed edges expressed as source/destination arrays."""

    src: npt.NDArray[np.int64]
    dst: npt.NDArray[np.int64]

    def to_numpy(self) -> npt.NDArray[np.int64]:
        """Return edges as a `(2, M)` NumPy array.

        Returns
        -------
        ndarray of int64, shape `(2, M)`
            Stacked `[src, dst]` edge-index array.

        """
        return np.array([self.src, self.dst], dtype=np.int64)


def get_edges_from_adj_list(
    adj_list: dict[int, list[int]], edge_type: EdgeType | None = None
) -> list[tuple[int, int, EdgeType]]:
    """Convert an adjacency-list dict into a flat list of edge tuples.

    Parameters
    ----------
    adj_list : dict[int, list[int]]
        Mapping from source node index to a list of destination indices.
    edge_type : EdgeType, optional
        Edge type to assign to every edge.  Defaults to
        `EdgeType.VIRTUAL`.

    Returns
    -------
    list[tuple[int, int, EdgeType]]
        List of `(src, dst, edge_type)` tuples.

    """
    if edge_type is None:
        edge_type = EdgeType.VIRTUAL

    edges: list[tuple[int, int, EdgeType]] = []
    for src, destinations in adj_list.items():
        edges.extend((src, dst, edge_type) for dst in destinations)
    return edges


def _normalize_edge_types(edge_type: EdgeType | Sequence[EdgeType], n_edges: int) -> list[EdgeType]:
    """Convert a single EdgeType or Sequence into a validated list of EdgeTypes."""
    if isinstance(edge_type, EdgeType):
        return [edge_type] * n_edges

    edge_types = list(edge_type)
    if len(edge_types) != n_edges:
        msg = (
            f"Length of edge_type must equal the number of edges. "
            f"Got {len(edge_types)} for {n_edges} edges."
        )
        raise ValueError(msg)

    return edge_types
