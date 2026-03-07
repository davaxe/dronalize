from dronalize.builders.graph_builder_osm import OSMMapGraphBuilder


class InDGraphBuilder(OSMMapGraphBuilder):
    """Map graph builder for the inD dataset.

    The inD (intersections Drone) dataset was recorded at urban intersections in
    Germany. It contains naturalistic trajectories of various traffic
    participants — including cars, trucks, bicycles, and pedestrians — extracted
    from drone footage.

    This builder inherits all OSM-based graph construction logic from
    `OSMMapGraphBuilder`, as the inD dataset ships with lanelet2-based OSM map
    files that describe the road topology at each recorded location.

    """
