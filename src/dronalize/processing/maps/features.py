"""Semantic map feature primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from dronalize.core.categories import EdgeType

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
    interp_distance: float | None = None


MapFeature: TypeAlias = PointFeature | PathFeature


__all__ = ["MapFeature", "PathFeature", "Point", "PointFeature"]
