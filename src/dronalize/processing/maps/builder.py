"""Map graph building from semantic geometry features."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

from typing_extensions import override

from dronalize.processing.maps.compiler import (
    InterpolationStage,
    MapGraphCompiler,
    interpolate_position,
)
from dronalize.processing.maps.features import MapFeature, Point
from dronalize.processing.maps.options import MapBuildOptions

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from dronalize.core.categories import EdgeType
    from dronalize.core.maps import MapGraph


class MapBuilder(Protocol):
    """Minimal protocol for building a map graph."""

    def build(
        self, min_distance: float | None = None, interp_distance: float | None = None
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
        self, min_distance: float | None = None, interp_distance: float | None = None
    ) -> MapGraph:
        options = MapBuildOptions.from_distances(
            min_distance=min_distance, interp_distance=interp_distance, edge_remap=self.edge_remap()
        )
        compiler = MapGraphCompiler(options)
        return compiler.compile(self.iter_features())


def build_map(
    source: MapGeometrySource,
    *,
    min_distance: float | None = None,
    interp_distance: float | None = None,
) -> MapGraph:
    """Compile a geometry source directly into a `MapGraph`."""
    options = MapBuildOptions.from_distances(
        min_distance=min_distance, interp_distance=interp_distance, edge_remap=source.edge_remap()
    )
    compiler = MapGraphCompiler(options)
    return compiler.compile(source.iter_features())


__all__ = [
    "FeatureMapBuilder",
    "InterpolationStage",
    "MapBuilder",
    "MapGeometrySource",
    "Point",
    "build_map",
    "interpolate_position",
]
