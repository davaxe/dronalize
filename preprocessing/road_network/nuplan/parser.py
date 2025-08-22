from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Generic, Protocol, Self, TypeVar

from preprocessing.road_network.nuplan.geometry_parser import (
    GeoPackageBinaryHeader,
    WKBPayload,
    geometry_from_bytes,
)

if TYPE_CHECKING:
    from pathlib import Path


class NuplanMap:
    def __init__(self, gpkg_file: Path, map_meta_file: Path | None = None):
        self.gpkg_file = gpkg_file
        self.map_meta_file = map_meta_file


ID = TypeVar("ID", bound=Hashable)


@dataclass
class Geometry:
    header: GeoPackageBinaryHeader
    data: WKBPayload

    @classmethod
    def from_bytes(cls, data: bytes) -> Geometry:
        """Convert bytes data to a Geometry instance."""
        header, payload = geometry_from_bytes(data)
        return cls(header=header, data=payload)


@dataclass
class LayerBase(Protocol, Generic[ID]):
    id: ID
    geometry: Geometry

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> Self:
        """Convert a database row to an instance of the layer."""
        ...


class BoundaryType(IntEnum): ...


@dataclass
class Boundary(LayerBase):
    boundary_segment_id: int
    has_reflector: bool
    type: BoundaryType
    creator_id: str

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
