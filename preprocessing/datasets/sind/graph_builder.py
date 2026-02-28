from preprocessing.common.map.osm import OSMMapGraphBuilder
from preprocessing.common.plotting import plot_map_graph


class SindGraphBuilder(OSMMapGraphBuilder):
    """Map graph builder for the SIND dataset.

    This builder constructs a map graph from the OSM lanelet map files provided with the rounD
    dataset, inheriting all OSM parsing and graph construction logic from
    `OSMMapGraphBuilder`.
    """


if __name__ == "__main__":
    from pathlib import Path

    import altair as alt

    alt.renderers.enable("browser")
    path = Path("data/sind/data/location2.osm")

    builder = SindGraphBuilder(path)

    graph = builder.build()
    print(graph)
    plot_map_graph(graph).show()
