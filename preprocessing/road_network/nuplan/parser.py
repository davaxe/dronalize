"""Parser for NuPlan map data stored in GeoPackage format.

Only used data is currently parsed, and there is a lot more data
available in the GeoPackage!
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Protocol, Self, TypeVar

from pyproj import Transformer

from preprocessing.road_network.edge_type import EdgeType
from preprocessing.road_network.nuplan.geometry_parser import (
    GeoPackageBinaryHeader,
    WKBPayload,
    geometry_from_bytes,
)

if TYPE_CHECKING:
    from pathlib import Path


class NuPlanMap:
    def __init__(self, gpkg_file: Path, map_meta_file: Path | None = None):
        self.gpkg_file = gpkg_file
        self.map_meta_file = map_meta_file

        self.connection = sqlite3.connect(self.gpkg_file)

        # Read metadata
        cursor = self.connection.cursor()
        cursor.execute("SELECT key, value FROM meta")
        cursor = cursor.fetchall()
        self.metadata = dict(cursor)
        self.transform = Transformer.from_crs(
            self.metadata["geographicCoordSystem"],
            self.metadata["projectedCoordSystem"],
            always_xy=True,
        ).transform

    @cached_property
    def boundaries(self) -> dict[int, Boundary]:
        cursor = self.connection.cursor()
        cursor.execute(Boundary.sql())
        rows = cursor.fetchall()
        return {row[0]: Boundary.from_row(row) for row in rows}

    @cached_property
    def traffic_lights(self) -> dict[int, TrafficLight]:
        cursor = self.connection.cursor()
        cursor.execute(TrafficLight.sql())
        rows = cursor.fetchall()
        return {row[0]: TrafficLight.from_row(row) for row in rows}

    @cached_property
    def walkways(self) -> dict[int, WalkWay]:
        cursor = self.connection.cursor()
        cursor.execute(WalkWay.sql())
        rows = cursor.fetchall()
        return {row[0]: WalkWay.from_row(row) for row in rows}

    @cached_property
    def stop_polygons(self) -> dict[int, StopPolygon]:
        cursor = self.connection.cursor()
        cursor.execute(StopPolygon.sql())
        rows = cursor.fetchall()
        return {row[0]: StopPolygon.from_row(row) for row in rows}

    @cached_property
    def car_park_areas(self) -> dict[int, CarparkArea]:
        cursor = self.connection.cursor()
        cursor.execute(CarparkArea.sql())
        rows = cursor.fetchall()
        return {row[0]: CarparkArea.from_row(row) for row in rows}


ID = TypeVar("ID", bound=Hashable)


@dataclass(frozen=True, slots=True, repr=False)
class Geometry:
    header: GeoPackageBinaryHeader
    data: WKBPayload

    @classmethod
    def from_bytes(cls, data: bytes) -> Geometry:
        """Convert bytes data to a Geometry instance."""
        header, payload = geometry_from_bytes(data)
        return cls(header=header, data=payload)

    def __repr__(self) -> str:
        return f"Geometry(header={self.data.geometry_type.name}, n_bytes={len(self.data.payload)})"


@dataclass
class LayerBase(Protocol, Generic[ID]):
    id: ID
    geometry: Geometry

    @staticmethod
    def sql() -> str:
        """Return the SQL query for selecting a layer."""
        ...

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> Self:
        """Convert a database row to an instance of the layer."""
        ...

    def points_transform(
        self, transform: Callable[[float, float], tuple[float, float]]
    ) -> Iterator[tuple[float, float]]:
        """Transform the points of the layer geometry using the given transform function."""
        yield from self.geometry.data.points_transform(transform)

    def points(self) -> Iterator[tuple[float, float]]:
        """Yield the points of the layer geometry."""
        yield from self.geometry.data.points()

    def edge_type(self) -> EdgeType:
        """Return the edge type for the layer."""
        ...


class BoundaryType(IntEnum):
    """Types of boundaries in the NuPlan map.

    From: https://github.com/motional/nuplan-devkit/issues/219
    """

    LaneDivider = 0
    Unused = 1
    Roadside = 2
    Virtual = 3

    def edge_type(self) -> EdgeType:
        """Get the corresponding edge type for the boundary."""
        return _BOUNDARY_TYPE_TO_EDGE_TYPE.get(self, EdgeType.VIRTUAL)


_BOUNDARY_TYPE_TO_EDGE_TYPE: dict[BoundaryType, EdgeType] = {
    BoundaryType.LaneDivider: EdgeType.LINE_THIN,
    BoundaryType.Unused: EdgeType.VIRTUAL,
    BoundaryType.Roadside: EdgeType.ROAD_BORDER,
    BoundaryType.Virtual: EdgeType.VIRTUAL,
}


@dataclass
class Boundary(LayerBase):
    boundary_segment_id: int
    has_reflector: bool
    type: BoundaryType
    creator_id: str | None = None

    @staticmethod
    def sql() -> str:
        """Return the SQL query for selecting a boundary."""
        return "SELECT * FROM boundaries"

    @classmethod
    def from_row(cls, row: tuple[int, bytes, int, bool, int, str]) -> Boundary:
        _id, geom_data, boundary_segment_id, has_reflector, _type, creator_id = row
        return cls(
            id=_id,
            geometry=Geometry.from_bytes(geom_data),
            boundary_segment_id=boundary_segment_id,
            type=BoundaryType(_type),
            has_reflector=has_reflector,
            creator_id=creator_id,
        )

    def edge_type(self) -> EdgeType:
        """Get the corresponding edge type for the boundary."""
        return self.type.edge_type()


@dataclass
class WalkWay(LayerBase):
    creator_id: str | None = None

    @staticmethod
    def sql() -> str:
        """Return the SQL query for selecting a walkway."""
        return "SELECT * FROM walkways"

    @classmethod
    def from_row(cls, row: tuple[int, bytes, str | None]) -> WalkWay:
        _id, geom_data, creator_id = row
        return cls(
            id=_id,
            geometry=Geometry.from_bytes(geom_data),
            creator_id=creator_id,
        )

    def edge_type(self) -> EdgeType:
        return EdgeType.CURB


@dataclass
class TrafficLight(LayerBase):
    light_face_type: int
    creator_id: str | None = None

    @staticmethod
    def sql() -> str:
        """Return the SQL query for selecting a traffic light."""
        return (
            "SELECT fid, geom, light_face_type_fid, creator_id FROM traffic_lights"
        )

    @classmethod
    def from_row(
        cls,
        row: tuple[int, bytes, int, str | None],
    ) -> TrafficLight:
        (_id, geom_data, light_face_type, creator_id) = row
        return cls(
            id=_id,
            geometry=Geometry.from_bytes(geom_data),
            light_face_type=light_face_type,
            creator_id=creator_id,
        )

    def edge_type(self) -> EdgeType:
        """Return the edge type for the layer."""
        return EdgeType.REGULATORY


@dataclass
class StopPolygon(LayerBase):
    stop_polygon_type: int
    creator_id: str | None = None

    @staticmethod
    def sql() -> str:
        """Return the SQL query for selecting a stop polygon."""
        return (
            "SELECT fid, geom, stop_polygon_type_fid, creator_id FROM stop_polygons"
        )

    @classmethod
    def from_row(cls, row: tuple[int, bytes, int, str | None]) -> StopPolygon:
        _id, geom_data, stop_polygon_type, creator_id = row
        return cls(
            id=_id,
            geometry=Geometry.from_bytes(geom_data),
            stop_polygon_type=stop_polygon_type,
            creator_id=creator_id,
        )

    def edge_type(self) -> EdgeType:
        """Return the edge type for the layer."""
        return EdgeType.STOP


@dataclass
class CarparkArea(LayerBase):
    heading: float
    creator_id: str | None = None

    @staticmethod
    def sql() -> str:
        """Return the SQL query for selecting a car park area."""
        return "SELECT * FROM carpark_areas"

    @classmethod
    def from_row(cls, row: tuple[int, bytes, float, str | None]) -> CarparkArea:
        _id, geom_data, heading, creator_id = row
        return cls(
            id=_id,
            geometry=Geometry.from_bytes(geom_data),
            heading=heading,
            creator_id=creator_id,
        )

    def edge_type(self) -> EdgeType:
        """Return the edge type for the layer."""
        return EdgeType.VIRTUAL
