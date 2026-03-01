from collections.abc import Callable
from pathlib import Path

import utm
from osmium import SimpleHandler, osm
from typing_extensions import override

from preprocessing.core.datatypes.categories import EdgeType
from preprocessing.core.graph.builder import GraphBuilder
from preprocessing.core.graph.nodes import IntIDNode


class OSMMapGraphBuilder(SimpleHandler, GraphBuilder[int, IntIDNode]):
    """GraphBuilder implementation that constructs a MapGraph from OpenStreetMap (OSM) data.

    This is for instance useful for:
    - INTERACTION dataset,
    - or any other dataset where we want to leverage OSM data for map graph construction.

    """

    def __init__(
        self,
        osm_file: Path,
        utm_position_offset: tuple[float, float] = (0.0, 0.0),
        edge_type_mapping: Callable[[osm.Way], EdgeType] | None = None,
        *,
        include_edge_type_none: bool = False,
    ) -> None:
        """Initialize the OSMMapGraphBuilder.

        Parameters
        ----------
        osm_file : Path
            File path to the OSM data file (e.g., .osm or .pbf format).
        utm_position_offset : tuple[float, float], optional
            (x, y) offset to apply to all node positions after converting
            from lat/lon to UTM coordinates. Defaults to (0.0, 0.0).
        edge_type_mapping : Callable[[osm.Way], EdgeType], optional
            How to map the OSM way tags to EdgeType categories.
        include_edge_type_none : bool, optional
            Whether to include edges categorized as EdgeType.NONE based on
            the edge_type_mapping. Defaults to False (i.e., skip edges
            categorized as NONE).

        """
        if not osm_file.exists():
            msg = f"OSM file not found at {osm_file}. Please provide a valid path to the OSM data file."
            raise FileNotFoundError(msg)

        super().__init__()
        self._edge_type_mapping = edge_type_mapping or self._default_edge_type_mapping
        self._utm_position_offset = utm_position_offset
        self._osm_file = osm_file
        self._nodes: dict[int, IntIDNode] = {}
        self._include_edge_type_none = include_edge_type_none

    @staticmethod
    def _default_edge_type_mapping(way: osm.Way) -> EdgeType:
        return EdgeType.from_str(way.tags.get("type"), way.tags.get("subtype"))

    @override
    def new_node(self, x: float, y: float, z: float = 0) -> IntIDNode:
        return IntIDNode(x, y, z)

    @override
    def build_impl(
        self,
        min_distance: float | None = None,
        interp_distance: float | None = None,
    ) -> None:
        # Using `SimpleHandler.apply_file` will process the OSM file and call the corresponding
        # handler methods (`node`, `way`, `relation`) for each element in the file. This allows us
        # to build the graph incrementally as we parse the OSM data.
        # Note: `min_distance` and `interp_distance` are used implicitly in the `add_path_lazy` if the
        # `build` method is called with those parameters.
        self.apply_file(self._osm_file)

    def way(self, way: osm.Way) -> None:
        """Handle an OSM way."""
        nodes: list[IntIDNode] = [
            self._nodes[node.ref] for node in way.nodes if node.ref in self._nodes
        ]
        edge_type = self._edge_type_mapping(way)
        if edge_type == EdgeType.NONE and not self._include_edge_type_none:
            return

        self.add_path_lazy(nodes=nodes, edge_type=edge_type)

    def node(self, node: osm.Node) -> None:
        """Handle an OSM node."""
        x, y, _zone_number, _zone_letter = utm.from_latlon(node.location.lat, node.location.lon)
        x_offset, y_offset = self._utm_position_offset
        self._nodes[node.id] = self.new_node(float(x) + x_offset, float(y) + y_offset)
