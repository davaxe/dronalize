"""Map-graph builder for the OpenDD dataset."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from typing_extensions import Self, override

from dronalize.core.categories import EdgeType
from dronalize.processing.maps.builder import BaseMapBuilder, Point

if TYPE_CHECKING:
    from collections.abc import Iterable


class OpenDDMapBuilder(BaseMapBuilder):
    """A builder for creating a graph representation of an OpenDD map.

    Maps are stored in a SQLite database, typically located at:
    `rdbX/map_rdbX/map_rdbX.sqlite`, where `X` is the map number.

    More information about OpenDD can be found at:
    https://l3pilot.eu/data/opendd.html

    """

    def __init__(self, sqlite_map_path: Path) -> None:
        """Initialize the map builder with a SQLite file."""
        super().__init__()

        # Connect to the SQLite database
        if not sqlite_map_path.exists():
            msg = f"SQLite file {sqlite_map_path} does not exist."
            raise FileNotFoundError(msg)

        self.connection: sqlite3.Connection = sqlite3.connect(sqlite_map_path)
        self.cursor: sqlite3.Cursor = self.connection.cursor()

    @classmethod
    def from_sqlite_file(cls, sqlite_file: Path | str) -> OpenDDMapBuilder:
        """Create a map builder from a SQLite file.

        The file is by default located at: `rdbX/map_rdbX/map_rdbX.sqlite`, in
        the downloaded `OpenDD` data

        """
        if isinstance(sqlite_file, str):
            sqlite_file = Path(sqlite_file)

        return cls(sqlite_file)

    @override
    def build_impl(
        self, min_distance: float | None = None, interp_distance: float | None = None
    ) -> None:
        self._add_edges(min_distance, interp_distance)

    def _add_edges(self, min_distance: float | None, interp_distance: float | None) -> None:
        """Add edges to the graph based on the adjacency list."""
        _ = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row[0] for row in self.cursor.fetchall()]
        map_table = table_names[0]
        _ = self.cursor.execute(f"SELECT * FROM {map_table}")  # noqa: S608
        rows = self.cursor.fetchall()
        for row in rows:
            map_data_row = MapDataRow.try_from_row(row)
            if map_data_row is None:
                continue

            self._process_map_row(map_data_row, min_distance, interp_distance)

    def _process_map_row(
        self, map_data_row: MapDataRow, min_distance: float | None, interp_distance: float | None
    ) -> None:
        """Process a single map data row and update the graph."""
        self.add_node_edges_loop_min_dist(
            map_data_row.geometry.coordinates(),
            min_distance=min_distance,
            interp_distance=interp_distance,
            edge_type=map_data_row.edge_type(),
            is_polygon=isinstance(map_data_row.geometry, Polygon),
        )


class Geometry(Protocol):
    """Protocol for geometry types."""

    _coordinates: list[Point]

    @classmethod
    def from_str(cls, geometry_str: str) -> Self:
        """Create a Geometry object from a string representation."""
        ...

    def connections(self) -> Iterable[tuple[Point, Point]]:
        """Iterate over edge connections in the geometry."""
        ...

    def coordinates(self) -> list[Point]:
        """Get the coordinates of the geometry."""
        return self._coordinates


@dataclass
class LineString(Geometry):
    """A LineString geometry type."""

    _coordinates: list[Point]

    @classmethod
    @override
    def from_str(cls, geometry_str: str) -> LineString:
        """Create a LineString object from a string representation."""
        # Format: LINESTRING (x1 y1, x2 y2, ..., xn yn )
        if not geometry_str.startswith("LINESTRING"):
            msg = f"Expected 'LINESTRING', got '{geometry_str}'"
            raise ValueError(msg)

        coords_str = geometry_str[len("LINESTRING (") : -1]
        coords = coords_str.split(", ")
        return cls([(float(coord.split()[0]), float(coord.split()[1])) for coord in coords])

    @override
    def connections(self) -> Iterable[tuple[Point, Point]]:
        """Iterate over edge connections in the linestring."""
        for i in range(len(self._coordinates) - 1):
            yield (self._coordinates[i], self._coordinates[i + 1])


@dataclass(slots=True)
class Polygon(Geometry):
    """A Polygon geometry type."""

    _coordinates: list[Point]

    @classmethod
    @override
    def from_str(cls, geometry_str: str) -> Polygon:
        """Create a Polygon object from a string representation."""
        # Format: POLYGON ((x1 y1, x2 y2, ..., xn yn))
        if not geometry_str.startswith("POLYGON"):
            msg = f"Expected 'POLYGON', got '{geometry_str}'"
            raise ValueError(msg)

        coords_str = geometry_str[len("POLYGON ((") : -2]
        coords = coords_str.split(", ")
        points: list[Point] = [
            (float(coord.split()[0]), float(coord.split()[1])) for coord in coords[:-1]
        ]
        return cls(points)

    @override
    def connections(self) -> Iterable[tuple[Point, Point]]:
        """Iterate over edge connections in the polygon."""
        for i in range(len(self._coordinates)):
            yield (self._coordinates[i], self._coordinates[(i + 1) % len(self._coordinates)])


@dataclass
class MapDataRow:
    """A single row of map data from the OpenDD database."""

    index: int
    id: int
    type: str
    geometry: Geometry
    successors: list[int]
    predecessors: list[int]
    marking_type: str | None
    lane_type: str | None
    area_type: str | None

    @classmethod
    def try_from_row(cls, row: tuple[Any, ...]) -> MapDataRow | None:
        """Try to create a MapDataRow from a database row.

        Returns None if the row is invalid or missing required fields.
        """
        try:
            data = cls(
                row[0],
                row[1],
                row[2],
                MapDataRow._parse_geometry(row[3]),
                MapDataRow._parse_str_list(row[4]),
                MapDataRow._parse_str_list(row[5]),
                row[6],
                row[7],
                row[8],
            )
            if not data._is_valid():
                return None
        except (AttributeError, ValueError):
            return None

        return data

    def edge_type(self) -> EdgeType:
        """Get the edge type based on the lane type."""
        if self.marking_type is None:
            return EdgeType.VIRTUAL
        return _MARKING_TYPE_TO_EDGE_TYPE[self.marking_type]

    def _is_valid(self) -> bool:
        return self.area_type not in {"COMPLETE", "NEVER", "EMERGENCY"}

    @staticmethod
    def _parse_str_list(value: str | None) -> list[int]:
        """Parse a string representation of a list of integers."""
        if value == "[]" or value is None:
            return []
        value = value.strip("[]")
        return [int(float(x.strip())) for x in value.split(",")]

    @staticmethod
    def _parse_geometry(geometry_str: str) -> Geometry:
        """Parse a geometry string into a Geometry object."""
        if geometry_str.startswith("LINESTRING"):
            return LineString.from_str(geometry_str)

        if geometry_str.startswith("POLYGON"):
            return Polygon.from_str(geometry_str)
        # Add more geometry types as needed
        msg = f"Unsupported geometry type: {geometry_str}"
        raise ValueError(msg)


_MARKING_TYPE_TO_EDGE_TYPE: dict[str, EdgeType] = {
    "CURB": EdgeType.CURB,
    "CURB_TRAVERSABLE": EdgeType.CURB,
    "SHORT_DASHED_LINE": EdgeType.LINE_THIN_DASHED,
    "SINGLE_SOLID_LINE": EdgeType.LINE_THIN,
    "LONG_DASHED_LINE": EdgeType.LINE_THIN_DASHED,
    "NO_MARKING": EdgeType.VIRTUAL,
    "SHADED_AREA_MARKING": EdgeType.VIRTUAL,
    "GUARDRAIL": EdgeType.GUARD_RAIL,
}
