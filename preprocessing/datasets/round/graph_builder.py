from preprocessing.common.map.osm import OSMMapGraphBuilder


class RounDGraphBuilder(OSMMapGraphBuilder):
    """Map graph builder for the rounD dataset.

    The rounD (roundabouts Drone) dataset was recorded at roundabouts in Germany using drone
    footage. It contains naturalistic trajectories of various traffic participants — including
    cars, trucks, buses, motorcycles, bicycles, and pedestrians — navigating through roundabout
    intersections.

    This builder constructs a map graph from the OSM lanelet map files provided with the rounD
    dataset, inheriting all OSM parsing and graph construction logic from
    `OSMMapGraphBuilder`.
    """
