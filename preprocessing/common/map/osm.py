from collections.abc import Callable
from pathlib import Path

import osmium as osm
import utm
from osmium.osm import Node, Way
from typing_extensions import override

from preprocessing.common.map.plot import plot_map_graph
from preprocessing.core.graph.builder import GraphBuilder
from preprocessing.core.graph.nodes import IntIDNode
from preprocessing.core.datatypes.categories import EdgeType


class OSMMapGraphBuilder(osm.SimpleHandler, GraphBuilder[int, IntIDNode]):
    """GraphBuilder implementation that constructs a MapGraph from OpenStreetMap (OSM) data.

    This is for instance useful for:
    - INTERACTION dataset,
    - or any other dataset where we want to leverage OSM data for map graph construction.

    """

    def __init__(
        self,
        osm_file: Path,
        utm_position_offset: tuple[float, float] = (0.0, 0.0),
        edge_type_mapping: Callable[[Way], EdgeType] | None = None,
    ) -> None:
        """Initialize the OSMMapGraphBuilder.

        Args:
            osm_file: file path to the OSM data file (e.g., .osm or .pbf format).
            utm_position_offset: _(x, y) offset to apply to all node positions after converting
                from lat/lon to UTM coordinates. Defaults to (0.0, 0.0).
            edge_type_mapping: _How to map the OSM way tags to EdgeType categories.

        """
        if not osm_file.exists():
            msg = f"OSM file not found at {osm_file}. Please provide a valid path to the OSM data file."
            raise FileNotFoundError(msg)

        super().__init__()
        self._edge_type_mapping = edge_type_mapping or self._default_edge_type_mapping
        self._utm_position_offset = utm_position_offset
        self._osm_file = osm_file
        self._nodes: dict[int, IntIDNode] = {}

    @staticmethod
    def _default_edge_type_mapping(way: Way) -> EdgeType:
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

    def way(self, way: Way) -> None:
        """Handle a OSM way."""
        nodes: list[IntIDNode] = [
            self._nodes[node.ref] for node in way.nodes if node.ref in self._nodes
        ]
        edge_type = self._edge_type_mapping(way)
        self.add_path_lazy(nodes=nodes, edge_type=edge_type)

    def node(self, node: Node) -> None:
        """Handle OSM node."""
        x, y, _zone_number, _zone_letter = utm.from_latlon(node.location.lat, node.location.lon)
        x_offset, y_offset = self._utm_position_offset
        self._nodes[node.id] = self.new_node(float(x) + x_offset, float(y) + y_offset)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    builder = OSMMapGraphBuilder(Path("data/DR_CHN_Merging_ZS0.osm"))
    graph = builder.build(interp_distance=3, min_distance=1.5)
    plot_map_graph(graph, include_nodes=False)
    plt.show()
