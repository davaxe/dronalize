import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.datasets.common import utils
from dronalize.maps import BaseMapBuilder
from dronalize.maps.edge_type import EdgeType

if TYPE_CHECKING:
    from dronalize.maps.builder import Point


@dataclass
class OSMWay:
    """Lightweight representation of an OSM Way to replace osmium.Way."""

    tags: dict[str, str]


class OSMMapBuilder(BaseMapBuilder):
    """Map builder that constructs a `MapGraph` from OpenStreetMap (OSM) XML data."""

    def __init__(
        self,
        osm_file: Path,
        utm_position_offset: tuple[float, float] = (0.0, 0.0),
        edge_type_mapping: Callable[[OSMWay], EdgeType] | None = None,
        *,
        include_edge_type_none: bool = False,
    ) -> None:
        """Initialize the OSM map builder.

        Parameters
        ----------
        osm_file : Path
            File path to the OSM XML data file (.osm format).
        utm_position_offset : tuple[float, float], optional
            (x, y) offset to apply to all node positions after converting
            from lat/lon to UTM coordinates. Defaults to (0.0, 0.0).
        edge_type_mapping : Callable[[OSMWay], EdgeType], optional
            How to map the OSM way tags to EdgeType categories.
        include_edge_type_none : bool, optional
            Whether to include edges categorized as EdgeType.NONE based on
            the edge_type_mapping. Defaults to False.

        """
        if not osm_file.exists():
            msg = (
                f"OSM file not found at {osm_file}. "
                "Please provide a valid path to the OSM data file."
            )
            raise FileNotFoundError(msg)

        super().__init__()
        self._edge_type_mapping: Callable[[OSMWay], EdgeType] = (
            edge_type_mapping or self._default_edge_type_mapping
        )
        self._utm_position_offset: tuple[float, float] = utm_position_offset
        self._osm_file: Path = osm_file
        self._nodes: dict[int, Point] = {}
        self._include_edge_type_none: bool = include_edge_type_none

    @staticmethod
    def _default_edge_type_mapping(way: OSMWay) -> EdgeType:
        return EdgeType.from_str(way.tags.get("type"), way.tags.get("subtype"))

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

        x, y, _, _ = utils.from_latlon(lat, lon)
        self._nodes[node_id] = (x + x_offset, y + y_offset)

        # Clear element from memory once processed
        elem.clear()
        root.clear()

    def _process_way(self, elem: ET.Element, root: ET.Element) -> None:
        """Process an OSM way element."""
        points: list[Point] = []
        tags: dict[str, str] = {}

        # Extract node references and tags
        for child in elem:
            if child.tag == "nd":
                ref = int(child.attrib["ref"])
                if ref in self._nodes:
                    points.append(self._nodes[ref])
            elif child.tag == "tag":
                tags[child.attrib["k"]] = child.attrib["v"]

        way = OSMWay(tags=tags)
        edge_type = self._edge_type_mapping(way)

        if (edge_type != EdgeType.NONE or self._include_edge_type_none) and points:
            self.add_path_lazy(points=points, edge_type=edge_type)

        # Clear element from memory once processed
        elem.clear()
        root.clear()

    @override
    def build_impl(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        # Use iterparse for memory-efficient incremental XML parsing
        context = ET.iterparse(self._osm_file, events=("start", "end"))
        context = iter(context)
        _, root = next(context)

        x_offset, y_offset = self._utm_position_offset

        for event, elem in context:
            if event != "end":
                continue
            if elem.tag == "node":
                self._process_node(elem, x_offset, y_offset, root)
            elif elem.tag == "way":
                self._process_way(elem, root)
