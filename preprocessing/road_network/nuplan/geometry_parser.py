from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag, auto
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self

from torch import ge

if TYPE_CHECKING:
    from collections.abc import Iterator


def geometry_from_bytes(
    geom_data: bytes,
) -> tuple[GeoPackageBinaryHeader, WKBPayload]:
    """Load geometry data from bytes."""
    header = GeoPackageBinaryHeader.from_bytes(geom_data)
    payload = WKBPayload.from_bytes(geom_data[header.n_bytes() :])
    return header, payload


class Endianness(IntEnum):
    """Represents the endianness of the binary data."""

    LITTLE = 0
    BIG = 1

    def str_repr(self) -> Literal["little", "big"]:
        """Return a string representation of the endianness."""
        return "little" if self == Endianness.LITTLE else "big"

    def str_symbol(self) -> Literal["<", ">"]:
        """Return a string representation of the endianness symbol."""
        return "<" if self == Endianness.LITTLE else ">"


class GeometryType(IntEnum):
    """Represents the type of geometry."""

    POINT = 1
    LINESTRING = 2
    POLYGON = 3


class Dimension(IntEnum):
    XY = 2
    XYZ = 3
    XYM = 3
    XYZM = 4


class Envelope(IntEnum):
    """Represents the envelope (bounding box) of the geometry."""

    NONE = 0
    XY = 1
    XYZ = 2
    XYM = 3
    XYZM = 4

    def n_bytes(self) -> int:
        """Return the number of bytes used by the envelope.

        Returns:
            The number of bytes used by the envelope.

        """
        match self:
            case Envelope.NONE:
                return 0
            case Envelope.XY:
                return 4 * 8
            case Envelope.XYZ:
                return 6 * 8
            case Envelope.XYM:
                return 6 * 8
            case Envelope.XYZM:
                return 8 * 8

    def n_values(self) -> int:
        """Return the number of values (coordinates) in the envelope.

        Returns:
            The number of values in the envelope.

        """
        return self.n_bytes() // 8


class GeoPackageBinaryFlags(IntFlag):
    """Represents the flags used in the GeoPackage binary format."""

    BYTE_ORDER = 0b00000001
    ENVELOPE_MASK = 0b00001110
    EMPTY_GEOMETRY = 0b00010000
    EXTENDED = 0b00100000
    RESERVED = 0b11000000

    # Helpers for envelope values
    ENVELOPE_NONE = 0b00000000  # 0
    ENVELOPE_XY = 0b00000010  # 1 << 1
    ENVELOPE_XYZ = 0b00000100  # 2 << 1
    ENVELOPE_XYM = 0b00000110  # 3 << 1
    ENVELOPE_XYZM = 0b00001000  # 4 << 1

    def envelope(self) -> Envelope:
        """Extract the envelope code (0-4) from bits 1-3."""
        return Envelope((self & GeoPackageBinaryFlags.ENVELOPE_MASK) >> 1)

    def is_empty(self) -> bool:
        """Check if the geometry is empty."""
        return bool(self & GeoPackageBinaryFlags.EMPTY_GEOMETRY)

    def is_extended(self) -> bool:
        """Check if the geometry is extended."""
        return bool(self & GeoPackageBinaryFlags.EXTENDED)

    def is_little_endian(self) -> bool:
        """Check if the geometry is little endian."""
        return bool(self & GeoPackageBinaryFlags.BYTE_ORDER)


@dataclass(frozen=True, slots=True)
class GeoPackageBinaryHeader:
    """Represents the header of a GeoPackage binary file."""

    magic: str
    version: int
    flags: GeoPackageBinaryFlags
    srs_id: int
    envelope: list[float]

    def __post_init__(self) -> None:
        """Validate the header after initialization."""
        if self.magic != "GP":
            msg = f"Invalid magic number: {self.magic}"
            raise ValueError(msg)

    @classmethod
    def from_bytes(cls, data: bytes) -> GeoPackageBinaryHeader:
        """Deserialize from bytes.

        Args:
            data: The byte data to deserialize.

        Returns:
            A GeoPackageBinaryHeader instance.

        """
        flags = GeoPackageBinaryFlags(int.from_bytes(data[3:4], "little"))
        byteorder, byteorder_sym = (
            ("little", "<") if flags.is_little_endian() else ("big", ">")
        )
        envelope = flags.envelope()
        return cls(
            data[0:2].decode("ascii"),
            int.from_bytes(data[2:3], "little", signed=False),
            flags,
            int.from_bytes(data[4:8], byteorder),
            envelope=list(
                struct.unpack(
                    byteorder_sym + "d" * envelope.n_values(),
                    data[8 : 8 + envelope.n_bytes()],
                ),
            ),
        )

    def n_bytes(self) -> int:
        """Return the number of bytes used by the header."""
        return (
            2  # magic
            + 1  # version
            + 1  # flags
            + 4  # srs_id
            + len(self.envelope) * 8  # envelope
        )


@dataclass(frozen=True, slots=True)
class WKBPayload:
    """Represents the Well-Known Binary (WKB) payload."""

    endianness: Endianness
    geometry_type: GeometryType
    payload: bytes
    dim: Dimension

    @classmethod
    def from_bytes(cls, data: bytes) -> WKBPayload:
        """Deserialize from bytes.

        Args:
            data: The byte data to deserialize.

        Returns:
            A WKBPayload instance.

        """
        endianness = GeometryParser.parse_byteorder(data)
        geometry_type_int, dim = GeometryParser.parse_geometry_type(
            data,
            byteorder=endianness,
            offset=1,
        )

        return cls(endianness, GeometryType(geometry_type_int), data, dim)

    def data(self) -> Iterator[tuple[float, float]]:
        """Extract data points from the payload."""
        parser = PARSER_MAP[self.geometry_type]
        yield from parser.parse(
            self.payload,
            byteorder=self.endianness,
            dim=self.dim,
        ).points()


@dataclass(frozen=True, slots=True)
class GeometryParser(Protocol):
    @staticmethod
    def parse_byteorder(data: bytes) -> Endianness:
        if data[0] == 1:
            return Endianness.LITTLE
        return Endianness.BIG

    @staticmethod
    def parse_geometry_type(
        data: bytes,
        *,
        byteorder: Endianness,
        offset: int = 0,
    ) -> tuple[GeometryType, Dimension]:
        geometry_type_int = int.from_bytes(
            data[offset : offset + 4],
            byteorder.str_repr(),
            signed=False,
        )
        if geometry_type_int > 3000:
            dim = Dimension.XYZM
            geometry_type_int -= 3000
        elif geometry_type_int > 2000:
            dim = Dimension.XYM
            geometry_type_int -= 2000
        elif geometry_type_int > 1000:
            dim = Dimension.XYZ
            geometry_type_int -= 1000
        else:
            dim = Dimension.XY

        if geometry_type_int not in {1, 2, 3}:
            msg = f"Unsupported geometry type: {geometry_type_int}, "
            msg += f"supported types are: {list(GeometryType)}"
            raise ValueError(msg)

        return GeometryType(geometry_type_int), dim

    @classmethod
    def parse(
        cls: type[Self],
        data: bytes,
        *,
        byteorder: Endianness,
        offset: int = 0,
        dim: Dimension = Dimension.XY,
    ) -> Self: ...

    def points(self) -> Iterator[tuple[float, float]]: ...

    def n_bytes(self) -> int: ...


@dataclass(frozen=True, slots=True)
class Point(GeometryParser):
    x: float
    y: float
    z: float | None = None
    m: float | None = None

    @classmethod
    def parse(
        cls,
        data: bytes,
        *,
        byteorder: Endianness,
        offset: int = 0,
        dim: Dimension = Dimension.XY,
    ) -> Self:
        return cls(
            *struct.unpack(
                f"{byteorder.str_symbol()}" + "d" * dim,
                data[offset : offset + 8 * dim],
            ),
        )

    def n_bytes(self) -> int:
        dim = 2
        dim += 1 if self.z is not None else 0
        dim += 1 if self.m is not None else 0
        return 8 * dim


@dataclass(frozen=True, slots=True)
class LinearRing(GeometryParser):
    points_list: list[Point]

    @classmethod
    def parse(
        cls,
        data: bytes,
        *,
        byteorder: Endianness,
        offset: int = 0,
        dim: Dimension = Dimension.XY,
    ) -> Self:
        num_points = int.from_bytes(
            data[offset : offset + 4],
            byteorder.str_repr(),
            signed=False,
        )
        points = []
        for i in range(num_points):
            point_data = data[offset + 4 + i * 16 : offset + 4 + (i + 1) * 16]
            points.append(Point.parse(point_data, byteorder=byteorder, dim=dim))
        return cls(points_list=points)

    def n_bytes(self) -> int:
        return 4 + sum(p.n_bytes() for p in self.points_list)

    def points(self) -> Iterator[tuple[float, float]]:
        return ((p.x, p.y) for p in self.points_list)


@dataclass(frozen=True, slots=True)
class WKBPoint(GeometryParser):
    byteorder: Endianness
    geometry_type: GeometryType
    point: Point

    @classmethod
    def parse(
        cls: type[Self],
        data: bytes,
        *,
        byteorder: Endianness,
        offset: int = 0,
        dim: Dimension = Dimension.XY,
    ) -> Self:
        byteorder = GeometryParser.parse_byteorder(data)
        geometry_type, _dim = GeometryParser.parse_geometry_type(
            data,
            byteorder=byteorder,
            offset=offset + 1,
        )

        if geometry_type != GeometryType.POINT:
            msg = f"Expected geometry type `POINT`, got {geometry_type}"
            raise ValueError(msg)
        if _dim != dim:
            msg = f"Dimension mismatch: expected {dim}, got {_dim}"
            raise ValueError(msg)

        point = Point.parse(data, byteorder=byteorder, offset=offset + 5, dim=dim)
        return cls(
            byteorder=byteorder,
            geometry_type=geometry_type,
            point=point,
        )

    def n_bytes(self) -> int:
        return 1 + 4 + 4 + self.point.n_bytes()

    def points(self) -> Iterator[tuple[float, float]]:
        yield (self.point.x, self.point.y)


@dataclass(frozen=True, slots=True)
class WKBLineString(GeometryParser):
    byteorder: Endianness
    geometry_type: GeometryType
    points_list: list[Point]

    @classmethod
    def parse(
        cls: type[Self],
        data: bytes,
        *,
        byteorder: Endianness,
        offset: int = 0,
        dim: Dimension = Dimension.XY,
    ) -> Self:
        byteorder = GeometryParser.parse_byteorder(data)
        geometry_type, _dim = GeometryParser.parse_geometry_type(
            data,
            byteorder=byteorder,
            offset=offset + 1,
        )

        if geometry_type != GeometryType.LINESTRING:
            msg = f"Expected geometry type `LINESTRING`, got {geometry_type}"
            raise ValueError(msg)

        num_points = int.from_bytes(
            data[offset + 5 : offset + 9],
            byteorder.str_repr(),
            signed=False,
        )
        points = []
        point_offset = offset + 9
        for _ in range(num_points):
            point = Point.parse(
                data,
                byteorder=byteorder,
                offset=point_offset,
                dim=dim,
            )
            points.append(point)
            point_offset += point.n_bytes()

        return cls(
            byteorder=byteorder,
            geometry_type=geometry_type,
            points_list=points,
        )

    def n_bytes(self) -> int:
        return 1 + 4 + 4 + sum(p.n_bytes() for p in self.points_list)

    def points(self) -> Iterator[tuple[float, float]]:
        return ((p.x, p.y) for p in self.points_list)


@dataclass(frozen=True, slots=True)
class WKBPolygon(GeometryParser):
    byteorder: Endianness
    geometry_type: GeometryType
    rings: list[LinearRing]

    @classmethod
    def parse(
        cls: type[Self],
        data: bytes,
        *,
        byteorder: Endianness,
        offset: int = 0,
        dim: Dimension = Dimension.XY,
    ) -> Self:
        byteorder = GeometryParser.parse_byteorder(data)
        geometry_type, _dim = GeometryParser.parse_geometry_type(
            data,
            byteorder=byteorder,
            offset=offset + 1,
        )

        if geometry_type != GeometryType.POLYGON:
            msg = f"Expected geometry type `POLYGON`, got {geometry_type}"
            raise ValueError(msg)

        num_rings = int.from_bytes(
            data[offset + 5 : offset + 9],
            byteorder.str_repr(),
            signed=False,
        )
        rings = []
        ring_offset = offset + 9
        for _ in range(num_rings):
            ring = LinearRing.parse(
                data,
                byteorder=byteorder,
                offset=ring_offset,
                dim=dim,
            )
            rings.append(ring)
            ring_offset += ring.n_bytes()

        return cls(
            byteorder=byteorder,
            geometry_type=geometry_type,
            rings=rings,
        )

    def n_bytes(self) -> int:
        return 1 + 4 + 4 + sum(ring.n_bytes() for ring in self.rings)

    def points(self) -> Iterator[tuple[float, float]]:
        for ring in self.rings:
            yield from ring.points()


PARSER_MAP: dict[GeometryType, type[GeometryParser]] = {
    GeometryType.POINT: WKBPoint,
    GeometryType.LINESTRING: WKBLineString,
    GeometryType.POLYGON: WKBPolygon,
}


if __name__ == "__main__":
    data_path = Path("data/maps/sg-one-north/9.17.1964/map.gpkg")
    database = sqlite3.connect(data_path)
    # Get table "boundaries" data "geom"
    cursor = database.cursor()
    cursor.execute("SELECT geom FROM traffic_lights")
    rows = cursor.fetchall()
    for geom_data, *_ in rows:
        header = GeoPackageBinaryHeader.from_bytes(geom_data)
        payload = WKBPayload.from_bytes(geom_data[header.n_bytes() :])
        for x, y in payload.data():
            print(x, y)
    database.close()
