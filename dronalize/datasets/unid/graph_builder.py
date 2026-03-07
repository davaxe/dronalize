from dronalize.builders.graph_builder_osm import OSMMapGraphBuilder


class UniDGraphBuilder(OSMMapGraphBuilder):
    """GraphBuilder implementation for the uniD dataset.

    The uniD dataset was recorded at urban intersections in Germany using drone
    footage. It contains naturalistic trajectories of various traffic
    participants, including cars, trucks, buses, motorcycles, bicycles, and
    pedestrians navigating through signalised and unsignalised intersections.

    This builder inherits all OSM-based map graph construction logic from
    `OSMMapGraphBuilder`, as the uniD dataset ships with lanelet2-based OSM map
    files that describe the road topology of each recorded location.

    """
