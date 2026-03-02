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


if __name__ == "__main__":
    import time
    from pathlib import Path

    import altair as alt

    from preprocessing.common.plotting import plot_map_graph

    alt.renderers.enable("browser")
    graph_builder = ExiDGraphBuilder(
        Path(
            "/home/west/Developer/behavior-prediction/datasets/exiD/maps/lanelets/0_cologne_butzweiler/location0.osm"
        )
    )
    start = time.perf_counter()
    graph = graph_builder.build()
    print(f"Graph built in {time.perf_counter() - start:.2f} seconds")
    plot_map_graph(graph).show()
