from preprocessing.common.map_utils.osm import OSMMapGraphBuilder
from preprocessing.common.map_utils.plot import plot_map_graph


class ExiDGraphBuilder(OSMMapGraphBuilder):
    """Map graph builder for the exiD dataset.

    The exiD (extracted from Drone) dataset was recorded at highway exits and entries in Germany.
    It contains naturalistic vehicle trajectories extracted from drone footage, covering a variety
    of traffic participants including cars, trucks, buses, and motorcycles interacting at
    on-ramps and off-ramps.

    This builder constructs a map graph from the OSM lanelet map files provided with the exiD
    dataset, inheriting all OSM parsing and graph construction logic from
    `OSMMapGraphBuilder`.
    """


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    path = Path(
        "/home/west/Developer/behavior-prediction/datasets/exiD/maps/lanelets/0_cologne_butzweiler/location0.osm"
    )
    graph_builder = ExiDGraphBuilder(path)
    map_graph = graph_builder.build()
    plot_map_graph(map_graph)
    plt.show()
