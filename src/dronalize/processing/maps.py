"""Public map-processing API and semantic map compilation helpers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
from math import ceil
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
from typing_extensions import override

from dronalize.core.categories import EdgeType
from dronalize.core.maps import MapGraph

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from dronalize.core.scene.model import Scene


Point: TypeAlias = tuple[float, float]
"""A 2-D point as `(x, y)`."""


@dataclass(frozen=True, slots=True)
class PointFeature:
    """A standalone map point."""

    point: Point


@dataclass(frozen=True, slots=True)
class PathFeature:
    """A polyline or polygon map feature."""

    points: tuple[Point, ...]
    edge_types: EdgeType | tuple[EdgeType, ...]
    closed: bool = False
    key: str | None = None
    min_distance: float | None = None
    interpolation_distance: float | None = None


MapFeature: TypeAlias = PointFeature | PathFeature


@dataclass(frozen=True, slots=True)
class MapBuildOptions:
    """Global sampling and edge-remapping options for map compilation."""

    min_distance: float
    interpolation_distance: float
    edge_remap: dict[EdgeType, EdgeType] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate distance parameters after initialization."""
        if self.interpolation_distance <= 0.0:
            msg = "interpolation_distance must be greater than 0."
            raise ValueError(msg)
        if not (0.0 <= self.min_distance <= self.interpolation_distance):
            msg = (
                "min_distance must be in the range "
                f"[0, interpolation_distance] ([0, {self.interpolation_distance}])."
            )
            raise ValueError(msg)

    @classmethod
    def from_distances(
        cls,
        min_distance: float | None,
        interpolation_distance: float | None,
        *,
        edge_remap: Mapping[EdgeType, EdgeType] | None = None,
    ) -> MapBuildOptions:
        """Create validated options from nullable distance inputs."""
        resolved_min_distance = 0.0 if min_distance is None else min_distance
        resolved_interpolation_distance = (
            np.inf if interpolation_distance is None else interpolation_distance
        )
        return cls(
            min_distance=resolved_min_distance,
            interpolation_distance=resolved_interpolation_distance,
            edge_remap={} if edge_remap is None else dict(edge_remap),
        )


class InterpolationStage(IntEnum):
    """Stage of an interpolation step."""

    INTERMEDIATE = auto()
    LAST = auto()


def interpolate_position(
    src: Point, dst: Point, target_distance: float | None = None
) -> list[tuple[InterpolationStage, int, Point]]:
    """Interpolate positions between *src* and *dst*."""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    distance = math.sqrt(dx * dx + dy * dy)

    if target_distance is None or target_distance == float("inf") or distance <= target_distance:
        return [(InterpolationStage.LAST, 0, dst)]

    num_steps = ceil(distance / target_distance)
    step_x = dx / num_steps
    step_y = dy / num_steps

    points: list[tuple[InterpolationStage, int, Point]] = []
    for i in range(1, num_steps):
        x = src[0] + step_x * i
        y = src[1] + step_y * i
        points.append((InterpolationStage.INTERMEDIATE, i - 1, (x, y)))

    points.append((InterpolationStage.LAST, num_steps - 1, dst))
    return points


@dataclass(frozen=True, slots=True)
class _CompiledPath:
    start_node: int
    end_node: int


@dataclass(slots=True)
class MapGraphCompiler:
    """Compile semantic map features into a `MapGraph`."""

    options: MapBuildOptions
    _x: list[float] = field(default_factory=list, init=False)
    _y: list[float] = field(default_factory=list, init=False)
    _edge_src: list[int] = field(default_factory=list, init=False)
    _edge_dst: list[int] = field(default_factory=list, init=False)
    _edge_types: list[int] = field(default_factory=list, init=False)
    _seen_edges: set[tuple[int, int, int]] = field(default_factory=set, init=False)
    _paths_by_key: dict[str, _CompiledPath] = field(default_factory=dict, init=False)

    def compile(self, features: Iterable[MapFeature]) -> MapGraph:
        """Compile semantic features into a `MapGraph`."""
        for feature in features:
            if isinstance(feature, PointFeature):
                _ = self._add_node(*feature.point)
            else:
                self._compile_path(feature)

        node_positions = np.column_stack([
            np.array(self._x, dtype=np.float64),
            np.array(self._y, dtype=np.float64),
        ])
        edge_indices = np.array([self._edge_src, self._edge_dst], dtype=np.int32)
        edge_types = np.array(self._edge_types, dtype=np.int32)
        return MapGraph(
            edge_indices=edge_indices, node_positions=node_positions, edge_types=edge_types
        )

    def _compile_path(self, feature: PathFeature) -> None:
        points = list(feature.points)
        if not points:
            return

        edge_types = _normalize_edge_types(
            feature.edge_types, len(points) - 1 + int(feature.closed)
        )
        sampled_points, sampled_edge_types = _sample_path(
            points=points,
            edge_types=edge_types,
            closed=feature.closed,
            min_distance=self.options.min_distance,
        )
        if not sampled_points:
            return

        first_node = self._add_node(*sampled_points[0])
        prev_node = first_node
        prev_point = sampled_points[0]
        for segment_index, dst_point in enumerate(sampled_points[1:]):
            dst_node = self._add_node(*dst_point)
            self._add_interpolated_edge(
                src_id=prev_node,
                src_point=prev_point,
                dst_id=dst_node,
                dst_point=dst_point,
                interpolation_distance=self.options.interpolation_distance,
                edge_type=sampled_edge_types[segment_index],
            )
            prev_node = dst_node
            prev_point = dst_point

        end_node = prev_node
        if feature.closed and len(sampled_points) > 1:
            self._add_interpolated_edge(
                src_id=prev_node,
                src_point=prev_point,
                dst_id=first_node,
                dst_point=sampled_points[0],
                interpolation_distance=self.options.interpolation_distance,
                edge_type=sampled_edge_types[-1],
            )

        if feature.key is not None:
            if feature.key in self._paths_by_key:
                msg = f"Duplicate path key {feature.key!r}."
                raise ValueError(msg)
            self._paths_by_key[feature.key] = _CompiledPath(
                start_node=first_node, end_node=end_node
            )

    def _add_interpolated_edge(
        self,
        *,
        src_id: int,
        src_point: Point,
        dst_id: int,
        dst_point: Point,
        interpolation_distance: float,
        edge_type: EdgeType,
    ) -> None:
        prev_id = src_id
        for stage, _, point in interpolate_position(
            src_point, dst_point, target_distance=interpolation_distance
        ):
            new_id = dst_id if stage == InterpolationStage.LAST else self._add_node(*point)
            self._add_edge(prev_id, new_id, edge_type)
            prev_id = new_id

    def _add_node(self, x: float, y: float) -> int:
        node_id = len(self._x)
        self._x.append(x)
        self._y.append(y)
        return node_id

    def _add_edge(self, src: int, dst: int, edge_type: EdgeType) -> None:
        edge_type_val = self.options.edge_remap.get(edge_type, edge_type)
        edge_key = (src, dst, int(edge_type_val))
        if edge_key in self._seen_edges:
            return

        self._seen_edges.add(edge_key)
        self._edge_src.append(src)
        self._edge_dst.append(dst)
        self._edge_types.append(int(edge_type_val))


def _normalize_edge_types(
    edge_types: EdgeType | tuple[EdgeType, ...], n_edges: int
) -> list[EdgeType]:
    if isinstance(edge_types, EdgeType):
        return [edge_types] * n_edges

    normalized = list(edge_types)
    if len(normalized) != n_edges:
        msg = (
            "Length of edge_types must equal the number of edges. "
            f"Got {len(normalized)} for {n_edges} edges."
        )
        raise ValueError(msg)
    return normalized


def _sample_path(
    *, points: list[Point], edge_types: list[EdgeType], closed: bool, min_distance: float
) -> tuple[list[Point], list[EdgeType]]:
    if len(points) < 2:
        return points[:1], []

    if min_distance <= 0.0:
        sampled_points = list(points)
        sampled_edge_types = list(edge_types)
        return sampled_points, sampled_edge_types

    min_dist_sq = min_distance**2
    sampled_points = [points[0]]
    sampled_edge_types: list[EdgeType] = []
    prev_point = points[0]

    i, j = 0, 1
    while i < len(points) - 1:
        dst_point = points[j]
        if _distance_sq(prev_point, dst_point) < min_dist_sq and j < len(points) - 1:
            j += 1
            continue

        sampled_points.append(dst_point)
        sampled_edge_types.append(edge_types[i])
        prev_point = dst_point
        i = j
        j = i + 1

    if closed and len(sampled_points) > 1:
        sampled_edge_types.append(edge_types[-1])

    return sampled_points, sampled_edge_types


def _distance_sq(src: Point, dst: Point) -> float:
    return (src[0] - dst[0]) ** 2 + (src[1] - dst[1]) ** 2


class MapBuilder(Protocol):
    """Minimal protocol for building a map graph."""

    def build(
        self, min_distance: float | None = None, interpolation_distance: float | None = None
    ) -> MapGraph:
        """Build the final `MapGraph`."""
        ...


class MapGeometrySource(Protocol):
    """Semantic source for map geometry features."""

    def iter_features(self) -> Iterable[MapFeature]:
        """Yield semantic map features."""
        ...

    def edge_remap(self) -> Mapping[EdgeType, EdgeType]:
        """Return edge remapping applied during compilation."""
        ...


class FeatureMapBuilder(MapBuilder, MapGeometrySource, ABC):
    """Base class for map builders that emit semantic geometry features."""

    @abstractmethod
    @override
    def iter_features(self) -> Iterable[MapFeature]: ...

    @override
    def edge_remap(self) -> Mapping[EdgeType, EdgeType]:
        return {}

    @override
    def build(
        self, min_distance: float | None = None, interpolation_distance: float | None = None
    ) -> MapGraph:
        options = MapBuildOptions.from_distances(
            min_distance=min_distance,
            interpolation_distance=interpolation_distance,
            edge_remap=self.edge_remap(),
        )
        compiler = MapGraphCompiler(options)
        return compiler.compile(self.iter_features())


def build_map(
    source: MapGeometrySource,
    *,
    min_distance: float | None = None,
    interpolation_distance: float | None = None,
) -> MapGraph:
    """Compile a geometry source directly into a `MapGraph`."""
    options = MapBuildOptions.from_distances(
        min_distance=min_distance,
        interpolation_distance=interpolation_distance,
        edge_remap=source.edge_remap(),
    )
    compiler = MapGraphCompiler(options)
    return compiler.compile(source.iter_features())


MapKey = str | None
"""Stable identifier for a map associated with a scene or source.

This alias mirrors [`dronalize.core.scene.MapKey`][] so runtime map helpers can
depend on the processing package without importing higher-level scene APIs.
"""

MapResolver = Callable[["Scene"], MapGraph | None]
"""Callable signature for lazily resolving a map graph for a scene.

Processing loaders attach resolvers to scenes so map materialization can be
deferred until a downstream consumer actually needs the graph.
"""


def no_map() -> MapResolver:
    """Create a resolver for datasets that do not expose map data.

    Returns
    -------
    MapResolver
        Resolver that always returns `None`.

    """

    def _resolve(_scene: Scene) -> None:
        return None

    _resolve.__name__ = "no_map"
    return _resolve


def shared_map(
    shared_name: dict[MapKey, str] | str, f: Callable[[Scene, MapGraph], MapGraph] | None = None
) -> MapResolver:
    """Create a resolver that materializes a scene map from shared memory.

    Parameters
    ----------
    shared_name : dict[MapKey, str] | str
        Shared-memory name or lookup table keyed by `scene.map_key`.
    f : Callable[[Scene, MapGraph], MapGraph] | None
        A function to apply to the map graph before returning it.
        If `None`, the map graph is returned as-is.

    Returns
    -------
    MapResolver
        Resolver that opens the shared-memory map, optionally applies `f`,
        and returns a detached copy.

    """

    def _resolve(scene: Scene) -> MapGraph | None:
        name = shared_name.get(scene.map_key) if isinstance(shared_name, dict) else shared_name
        if name is None:
            return None

        with MapGraph.from_shared(name) as map_graph:
            if f is None:
                return map_graph.copy()

            return f(scene, map_graph).copy()

    _resolve.__name__ = "shared_map"
    return _resolve
