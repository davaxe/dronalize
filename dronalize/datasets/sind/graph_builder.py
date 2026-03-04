from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import utm
from typing_extensions import override

from dronalize.common.map.osm import OSMMapGraphBuilder

if TYPE_CHECKING:
    from pathlib import Path


class SindGraphBuilder(OSMMapGraphBuilder):
    """Map graph builder for the SIND dataset.

    This builder constructs a map graph from the OSM lanelet map files provided
    with the SIND dataset. It dynamically finds the local origin in the OSM file
    and uses UTM projection offsets to build a local metric coordinate system.

    """

    def __init__(
        self,
        osm_file: Path,
        position_offset: tuple[float, float] = (0.0, 0.0),
        *,
        include_edge_type_none: bool = False,
    ) -> None:
        """Initialize the graph builder.

        Parameters
        ----------
        osm_file : Path
            Path to the OSM lanelet map file for the SIND dataset.
        position_offset : tuple[float, float], optional
            (x, y) offset to apply to all node positions after converting to
            local coordinates.
        include_edge_type_none : bool, optional
            Whether to include edges with type 'none' in the graph.

        """
        self._origin_lat, self._origin_lon = self._extract_origin(osm_file)
        (self._origin_utm_x, self._origin_utm_y, self._zone_number, self._zone_letter) = (
            utm.from_latlon(self._origin_lat, self._origin_lon)
        )
        super().__init__(osm_file, position_offset, include_edge_type_none=include_edge_type_none)

    @staticmethod
    def _extract_origin(osm_file: Path) -> tuple[float, float]:
        """Parse the XML sequentially to find the node tagged as 'origin'."""
        context = ET.iterparse(osm_file, events=("start", "end"))
        current_node_element = None

        for event, elem in context:
            if event == "start" and elem.tag == "node":
                current_node_element = elem
            elif event == "end" and elem.tag == "tag" and current_node_element is not None:
                if elem.attrib.get("k") == "name" and elem.attrib.get("v") == "origin":
                    lat = float(current_node_element.attrib["lat"])
                    lon = float(current_node_element.attrib["lon"])
                    return lat, lon
            elif event == "end" and elem.tag == "node":
                current_node_element = None
                elem.clear()  # Free memory

        # Default fallback if the origin tag is missing
        return 0.0, 0.0

    @override
    def _process_node(
        self,
        elem: ET.Element,
        x_offset: float,
        y_offset: float,
        root: ET.Element,
    ) -> None:
        """Process an OSM node element."""
        node_id = int(elem.attrib["id"])
        lat = float(elem.attrib["lat"])
        lon = float(elem.attrib["lon"])

        # Apply SIND-specific UTM projection
        x_utm, y_utm, _, _ = utm.from_latlon(
            lat,
            lon,
            force_zone_number=self._zone_number,
            force_zone_letter=self._zone_letter,
        )
        local_x = x_utm - self._origin_utm_x
        local_y = y_utm - self._origin_utm_y

        self._nodes[node_id] = (float(local_x) + x_offset, float(local_y) + y_offset)

        # Clear element from memory once processed
        elem.clear()
        root.clear()
