from dronalize.datasets.shared.osm_builder import OSMMapBuilder


class InDMapBuilder(OSMMapBuilder):
    """Map builder for the inD dataset.

    The inD (intersections Drone) dataset was recorded at urban intersections in
    Germany. It contains naturalistic trajectories of various traffic
    participants — including cars, trucks, bicycles, and pedestrians — extracted
    from drone footage.

    This builder inherits all OSM-based graph construction logic from
    `OSMMapBuilder`, as the inD dataset ships with lanelet2-based OSM map
    files that describe the road topology at each recorded location.

    """
