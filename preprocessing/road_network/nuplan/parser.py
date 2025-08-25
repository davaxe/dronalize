"""Parser for NuPlan map data stored in GeoPackage format.

Only used data is currently parsed, and there is a lot more data
available in the GeoPackage!
"""

from __future__ import annotations

import sqlite3
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import shapely.geometry

from preprocessing.road_network.edge_type import EdgeType

if TYPE_CHECKING:
    from collections.abc import Iterator


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

    def boundaries(
        self,
    ) -> Iterator[tuple[EdgeType, Iterator[tuple[float, float]]]]:
        data = gpd.read_file(self.gpkg_file, layer="boundaries")
        data.to_crs(crs=self.metadata["projectedCoordSystem"], inplace=True)
        for _, _, b_type, _, geom in data.itertuples(
            index=False,
            name=None,
        ):
            yield BoundaryType(b_type).edge_type(), _GET_POINTS[type(geom)](geom)

    def walkways(self) -> Iterator[tuple[EdgeType, Iterator[tuple[float, float]]]]:
        data = gpd.read_file(self.gpkg_file, layer="walkways")
        data.to_crs(crs=self.metadata["projectedCoordSystem"], inplace=True)
        for _, geom in data.itertuples(
            index=False,
            name=None,
        ):
            yield EdgeType.CURB, _GET_POINTS[type(geom)](geom)

    def traffic_lights(
        self,
    ) -> Iterator[tuple[EdgeType, Iterator[tuple[float, float]]]]:
        data = gpd.read_file(self.gpkg_file, layer="traffic_lights")
        data.to_crs(crs=self.metadata["projectedCoordSystem"], inplace=True)
        for row in data.itertuples(index=False, name=None):
            geom = row[-1]
            yield EdgeType.REGULATORY, _GET_POINTS[type(geom)](geom)

    def stop_polygons(
        self,
    ) -> Iterator[tuple[EdgeType, Iterator[tuple[float, float]]]]:
        data = gpd.read_file(self.gpkg_file, layer="stop_polygons")
        data.to_crs(crs=self.metadata["projectedCoordSystem"], inplace=True)
        for row in data.itertuples(index=False, name=None):
            geom = row[-1]
            yield EdgeType.STOP, _GET_POINTS[type(geom)](geom)

    def carpark_areas(
        self,
    ) -> Iterator[tuple[EdgeType, Iterator[tuple[float, float]]]]:
        data = gpd.read_file(self.gpkg_file, layer="carpark_areas")
        data.to_crs(crs=self.metadata["projectedCoordSystem"], inplace=True)
        for row in data.itertuples(index=False, name=None):
            geom = row[-1]
            yield EdgeType.VIRTUAL, _GET_POINTS[type(geom)](geom)


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


def get_points_linestring(
    linestring: shapely.geometry.LineString,
) -> Iterator[tuple[float, float]]:
    """Get the points from a LineString geometry."""
    return ((x, y) for x, y in linestring.coords)


def get_points_polygon(
    polygon: shapely.geometry.Polygon,
) -> Iterator[tuple[float, float]]:
    """Get the points from a Polygon geometry."""
    return ((x, y) for x, y in polygon.exterior.coords)


_GET_POINTS = {
    shapely.geometry.LineString: get_points_linestring,
    shapely.geometry.Polygon: get_points_polygon,
    shapely.geometry.Point: lambda point: iter([(point.x, point.y)]),
}

if __name__ == "__main__":
    path = "data/maps/sg-one-north/9.17.1964/map.gpkg"
    map_data = NuPlanMap(Path(path))
    for _type, boundary in map_data.carpark_areas():
        for point in boundary:
            print(point)
        break
