"""Semantic map feature primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from dronalize.core.categories import EdgeType

Point: TypeAlias = tuple[float, float]
"""A 2-D point as `(x, y)`."""

EndpointName: TypeAlias = Literal["start", "end"]


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


@dataclass(frozen=True, slots=True)
class EndpointLinkFeature:
    """A semantic link between endpoints of keyed path features."""

    src_key: str
    dst_key: str
    edge_type: EdgeType
    src_endpoint: EndpointName = "end"
    dst_endpoint: EndpointName = "start"
    max_distance: float | None = None


MapFeature: TypeAlias = PointFeature | PathFeature | EndpointLinkFeature


__all__ = [
    "EndpointLinkFeature",
    "EndpointName",
    "MapFeature",
    "PathFeature",
    "Point",
    "PointFeature",
]
