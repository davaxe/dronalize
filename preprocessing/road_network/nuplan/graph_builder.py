from __future__ import annotations

from pathlib import Path

from preprocessing.road_network.common import (
    GraphBuilder,
    IntIDNode,
    MapGraph,
)
from preprocessing.road_network.nuplan import parser


class NuScenesMapGraphBuilder(GraphBuilder[str, IntIDNode]):
    """A builder for creating a MapGraph from a NuscenesMap."""

    def __init__(self, nuscenes_map: parser.NuPlanMap) -> None:
        """Initialize the graph builder with a NuscenesMap."""
        super().__init__()
        self.map: parser.NuPlanMap = nuscenes_map

        self.edge_type_methods = {
            "boundary": self._add_boundary_edges,
            "walkway": self._add_walkway_edges,
            "traffic_light": self._add_traffic_light_edges,
            "stop_polygon": self._add_stop_polygon_edges,
            "carpark": self._add_carpark_area_edges,
        }

    @classmethod
    def from_data_file(
        cls,
        data_file: Path,
        map_meta_file: Path | None = None,
    ) -> NuScenesMapGraphBuilder:
        """Create a NuScenesMapGraphBuilder from a data file."""
        nuscenes_map = parser.NuPlanMap(data_file, map_meta_file)
        return cls(nuscenes_map)

    def build(
        self,
        *,
        interp_distance: float | None = None,
        ignore_edge_types: set[str] | None = None,
    ) -> MapGraph:
        """Build a `MapGraph` from the `NuPlanMap`.

        Args:
            interp_distance: the target distance for interpolation. If None,
                no interpolation is performed.
            ignore_edge_types: a set of edge types to ignore when building the
                graph.

        Returns:
            A `MapGraph` object containing the node positions, edge indices,
            and edge types.

        """
        if ignore_edge_types is None:
            ignore_edge_types = set()

        for edge_type, method in self.edge_type_methods.items():
            if edge_type not in ignore_edge_types:
                method(interp_distance=interp_distance)

        return self.build_graph(include_extra_nodes=True)

    def new_node(self, x: float, y: float, z: float = 0) -> IntIDNode:
        """Create a new node in the graph with given coordinates."""
        return IntIDNode(*self.map.transform(x, y), z=z)

    def _add_boundary_edges(self, interp_distance: float | None = None) -> None:
        for boundary in self.map.boundaries.values():
            self.add_node_edges_loop(
                [self.new_node(x, y) for x, y in boundary.points()],
                edge_type=boundary.edge_type(),
                interp_distance=interp_distance,
            )

    def _add_walkway_edges(self, interp_distance: float | None = None) -> None:
        for walkway in self.map.walkways.values():
            self.add_node_edges_loop(
                [self.new_node(x, y) for x, y in walkway.points()],
                edge_type=walkway.edge_type(),
                interp_distance=interp_distance,
            )

    def _add_traffic_light_edges(
        self,
        interp_distance: float | None = None,  # noqa: ARG002
    ) -> None:
        for traffic_light in self.map.traffic_lights.values():
            for x, y in traffic_light.points():
                self.add_extra_node(self.new_node(x, y))

    def _add_stop_polygon_edges(
        self,
        interp_distance: float | None = None,
    ) -> None:
        for stop_polygon in self.map.stop_polygons.values():
            self.add_node_edges_loop(
                [self.new_node(x, y) for x, y in stop_polygon.points()],
                edge_type=stop_polygon.edge_type(),
                interp_distance=interp_distance,
            )

    def _add_carpark_area_edges(
        self,
        interp_distance: float | None = None,
    ) -> None:
        for car_park_area in self.map.car_park_areas.values():
            self.add_node_edges_loop(
                [self.new_node(x, y) for x, y in car_park_area.points()],
                edge_type=car_park_area.edge_type(),
                interp_distance=interp_distance,
            )


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("qt5agg")

    map_file = Path("data/maps/us-pa-pittsburgh-hazelwood/9.17.1937/map.gpkg")
    map_builder = NuScenesMapGraphBuilder.from_data_file(map_file)
    map_graph = map_builder.build()

    plt.figure()
    positions = map_graph.node_positions.numpy()
    plt.scatter(positions[:, 0], positions[:, 1], s=1, c="red")
    plt.axis("equal")
    plt.show()
