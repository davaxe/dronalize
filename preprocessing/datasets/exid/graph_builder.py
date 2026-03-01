from preprocessing.common.map.osm import OSMMapGraphBuilder


class ExiDGraphBuilder(OSMMapGraphBuilder):
    """Map graph builder for the exiD dataset.

    The exiD (extracted from Drone) dataset was recorded at highway exits and
    entries in Germany. It contains naturalistic vehicle trajectories extracted
    from drone footage, covering a variety of traffic participants including
    cars, trucks, buses, and motorcycles interacting at on-ramps and off-ramps.

    This builder constructs a map graph from the OSM lanelet map files provided
    with the exiD dataset, inheriting all OSM parsing and graph construction
    logic from `OSMMapGraphBuilder`.

    """
