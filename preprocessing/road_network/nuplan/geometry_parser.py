from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

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
        """Return a string representation of the endianness.

        Can be used in some standard python contexts, e.g, `int.from_bytes`.

        Returns:
            A string indicating the endianness ("little" or "big").

        """
        return "little" if self == Endianness.LITTLE else "big"


class GeometryType(IntEnum):
    """Represents the type of geometry."""

    GEOMETRY = 0
    POINT = 1
    LINESTRING = 2
    POLYGON = 3
    MULTIPOINT = 4
    MULTILINESTRING = 5
    MULTIPOLYGON = 6
    GEOMETRYCOLLECTION = 7
    CIRCULAR_STRING = 8
    COMPOUND_CURVE = 9
    CURVE_POLYGON = 10
    MULTI_CURVE = 11
    MULTI_SURFACE = 12
    CURVE = 13
    SURFACE = 14


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


@dataclass
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


@dataclass
class WKBPayload:
    """Represents the Well-Known Binary (WKB) payload."""

    endianness: Endianness
    geometry_type: GeometryType
    payload: bytes
    has_z: bool
    has_m: bool

    @classmethod
    def from_bytes(cls, data: bytes) -> WKBPayload:
        """Deserialize from bytes.

        Args:
            data: The byte data to deserialize.

        Returns:
            A WKBPayload instance.

        """
        endianness = Endianness.LITTLE if data[0] == 1 else Endianness.BIG
        geometry_type_int = int.from_bytes(
            data[1:5],
            endianness.str_repr(),
            signed=False,
        )

        both_offset, m_offset, z_offset = 3000, 2000, 1000
        has_z, has_m = False, False
        if geometry_type_int > both_offset:
            has_z = True
            has_m = True
            geometry_type_int -= both_offset
        elif geometry_type_int > m_offset:
            has_z = False
            has_m = True
            geometry_type_int -= m_offset
        elif geometry_type_int > z_offset:
            has_z = False
            has_m = False
            geometry_type_int -= z_offset

        payload = data[5 + 4 :]
        return cls(
            endianness,
            GeometryType(geometry_type_int),
            payload,
            has_z,
            has_m,
        )

    @overload
    def data(
        self,
        *,
        include_z: Literal[False] = False,
        include_m: Literal[False] = False,
    ) -> Iterator[tuple[float, float]]: ...

    @overload
    def data(
        self,
        *,
        include_z: Literal[True],
        include_m: Literal[False] = False,
    ) -> Iterator[tuple[float, float, float]]: ...

    @overload
    def data(
        self,
        *,
        include_z: Literal[False] = False,
        include_m: Literal[True],
    ) -> Iterator[tuple[float, float, float]]: ...

    @overload
    def data(
        self,
        *,
        include_z: Literal[True],
        include_m: Literal[True],
    ) -> Iterator[tuple[float, float, float, float]]: ...

    def data(
        self,
        *,
        include_z: bool = False,
        include_m: bool = False,
    ) -> Iterator[tuple[float, ...]]:
        """Extract data points from the payload."""
        base_str = "dd"
        if include_z and self.has_z:
            base_str += "d"
        elif include_z and not self.has_z:
            msg = f"`include_z` was set to {include_z}, but `self.has_z` is {self.has_z}"
            raise ValueError(msg)
        if include_m and self.has_m:
            base_str += "d"
        elif include_m and not self.has_m:
            msg = f"`include_m` was set to {include_m}, but `self.has_m` is {self.has_m}"
            raise ValueError(msg)

        for point in struct.iter_unpack(
            (">" if self.endianness == Endianness.BIG else "<") + base_str,
            self.payload,
        ):
            yield point if include_z or include_m else point[:2]


if __name__ == "__main__":
    data_path = Path("data/maps/sg-one-north/9.17.1964/map.gpkg")
    database = sqlite3.connect(data_path)
    # Get table "boundaries" data "geom"
    cursor = database.cursor()
    cursor.execute("SELECT geom FROM boundaries")
    rows = cursor.fetchall()
    for geom_data, *_ in rows:
        header = GeoPackageBinaryHeader.from_bytes(geom_data)
        payload = WKBPayload.from_bytes(geom_data[header.n_bytes() :])
        for x, y in payload.data():
            print(x, y)
        break
    database.close()
