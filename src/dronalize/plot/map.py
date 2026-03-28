# pyright: standard

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final

import altair as alt  # Optional dependency, checked at import time (in __init__.py)
import polars as pl

from dronalize.core.maps.edge_types import EdgeType

if TYPE_CHECKING:
    from types import ModuleType

    from dronalize.core.maps.graph import MapGraph

Chart = alt.Chart | alt.LayerChart


def plot_map_graph(
    graph: MapGraph,
    width: int = 700,
    height: int = 700,
    alpha: float = 0.7,
    *,
    include_nodes: bool = False,
    disable_max_rows: bool = True,
    **kwargs: Any,  # noqa: ANN401
) -> Chart:
    """Plot a map graph using Altair.

    Parameters
    ----------
    graph : MapGraph
        The map graph to plot.
    width : int, optional
        Width of the chart in pixels. Defaults to 700.
    height : int, optional
        Height of the chart in pixels. Defaults to 700.
    alpha : float, optional
        Transparency level for the edges. Defaults to 0.7.
    include_nodes : bool, optional
        Whether to include node markers in the plot. Defaults to False.
    disable_max_rows : bool, optional
        Automatically bypass Altair's 5000 row limit. Defaults to True.
    **kwargs : Any
        Additional keyword arguments to pass to `alt.Chart.properties()`.

    Returns
    -------
    alt.Chart
        An Altair `alt.Chart` object.

    """
    if disable_max_rows:
        alt.data_transformers.disable_max_rows()

    if graph.num_edges == 0:
        return _empty_chart(alt, "Empty Graph")

    layers, edge_selection = _build_map_graph_layers(
        graph,
        alt=alt,
        alpha=alpha,
        include_nodes=include_nodes,
    )

    return (
        alt
        .layer(*layers)
        .add_params(edge_selection)
        .properties(width=width, height=height, title="Map Graph Visualization", **kwargs)
        .configure_view(stroke=None)
        .configure_title(fontSize=22)
        .configure_axis(labelFontSize=14, titleFontSize=16)
        .configure_legend(labelFontSize=14, titleFontSize=16)
        .interactive()
    )


def _empty_chart(alt: ModuleType, message: str) -> alt.Chart:
    """Create a simple text chart used for empty plotting inputs."""
    return alt.Chart(pl.DataFrame()).mark_text(size=16).encode(text=alt.value(message))


def _prepare_map_edge_dataframe(graph: MapGraph) -> pl.DataFrame:
    """Convert graph edges into a tabular representation for Altair."""
    start_indices = graph.edge_indices[0]
    end_indices = graph.edge_indices[1]

    start_pos = graph.node_positions[start_indices]
    end_pos = graph.node_positions[end_indices]

    return (
        pl
        .DataFrame({
            "x1": start_pos[:, 0],
            "y1": start_pos[:, 1],
            "x2": end_pos[:, 0],
            "y2": end_pos[:, 1],
            "edge_type_int": graph.edge_types,
        })
        .with_columns(
            pl
            .col("edge_type_int")
            .replace_strict(_ALL_EDGE_TYPES, default=EdgeType.VIRTUAL.name.title())
            .alias("edge_type"),
            ((pl.col("x1") - pl.col("x2")) ** 2 + (pl.col("y1") - pl.col("y2")) ** 2)
            .sqrt()
            .alias("length"),
        )
        .drop("edge_type_int")
    )


def _build_map_graph_layers(
    graph: MapGraph,
    *,
    alt: ModuleType,
    alpha: float,
    include_nodes: bool,
) -> tuple[list[alt.Chart], Any]:
    """Build the layered Altair charts for a map graph."""
    df_edges = _prepare_map_edge_dataframe(graph)

    color_dict, width_dict, dash_dict = _get_altair_style_dicts()

    present_types = df_edges["edge_type"].unique().to_list()
    present_types.sort()

    plot_domain = present_types
    plot_color = [color_dict.get(pt, "gray") for pt in present_types]
    plot_width = [width_dict.get(pt, 1.0) for pt in present_types]
    plot_dash = [dash_dict.get(pt, [1, 0]) for pt in present_types]

    axis_scale = alt.Scale(zero=False)
    edge_selection = alt.selection_point(fields=["edge_type"], bind="legend")

    lines = (
        alt
        .Chart(df_edges)
        .mark_rule()
        .encode(
            x=alt.X("x1", scale=axis_scale, title="X", axis=alt.Axis(grid=False)),
            y=alt.Y("y1", scale=axis_scale, title="Y", axis=alt.Axis(grid=False)),
            x2="x2",
            y2="y2",
            color=alt.Color(
                "edge_type:N",
                scale=alt.Scale(domain=plot_domain, range=plot_color),
                legend=alt.Legend(
                    title="Edge Type",
                    symbolType="stroke",
                    symbolStrokeWidth=3,
                    orient="right",
                ),
            ),
            size=alt.Size(
                "edge_type:N",
                scale=alt.Scale(domain=plot_domain, range=plot_width),
                legend=None,
            ),
            strokeDash=alt.StrokeDash(
                "edge_type:N",
                scale=alt.Scale(domain=plot_domain, range=plot_dash),
                legend=None,
            ),
            opacity=alt.when(edge_selection).then(alt.value(alpha)).otherwise(alt.value(0.05)),
            tooltip=[
                alt.Tooltip("edge_type:N", title="Type"),
                alt.Tooltip("x1:Q", title="Start X", format=".2f"),
                alt.Tooltip("y1:Q", title="Start Y", format=".2f"),
                alt.Tooltip("x2:Q", title="End X", format=".2f"),
                alt.Tooltip("y2:Q", title="End Y", format=".2f"),
                alt.Tooltip("length:Q", title="Length", format=".2f"),
            ],
        )
    )

    layers = [lines]

    if include_nodes:
        nodes_start = (
            alt
            .Chart(df_edges)
            .mark_circle(color="black", size=15)
            .encode(
                x=alt.X("x1", scale=axis_scale),
                y=alt.Y("y1", scale=axis_scale),
                opacity=alt.when(edge_selection).then(alt.value(1.0)).otherwise(alt.value(0.05)),
                tooltip=[
                    alt.Tooltip("x1:Q", title="Node X", format=".2f"),
                    alt.Tooltip("y1:Q", title="Node Y", format=".2f"),
                ],
            )
        )
        nodes_end = (
            alt
            .Chart(df_edges)
            .mark_circle(color="black", size=15)
            .encode(
                x=alt.X("x2", scale=axis_scale),
                y=alt.Y("y2", scale=axis_scale),
                opacity=alt.when(edge_selection).then(alt.value(1.0)).otherwise(alt.value(0.05)),
                tooltip=[
                    alt.Tooltip("x2:Q", title="Node X", format=".2f"),
                    alt.Tooltip("y2:Q", title="Node Y", format=".2f"),
                ],
            )
        )
        layers.extend([nodes_start, nodes_end])

    return layers, edge_selection


_STYLE_MAP: Final[dict[int, dict]] = {
    EdgeType.NONE.value: {"color": "gray", "width": 0.5, "dash": [1, 0]},
    EdgeType.ROAD_BORDER.value: {"color": "black", "width": 2.0, "dash": [1, 0]},
    EdgeType.CURB.value: {"color": "black", "width": 2.5, "dash": [1, 0]},
    EdgeType.REGULATORY.value: {"color": "red", "width": 1.0, "dash": [1, 0]},
    EdgeType.VIRTUAL.value: {"color": "blue", "width": 0.5, "dash": [2, 2]},
    EdgeType.LINE_THIN.value: {"color": "gray", "width": 1.0, "dash": [1, 0]},
    EdgeType.LINE_THIN_DASHED.value: {"color": "gray", "width": 1.0, "dash": [5, 5]},
    EdgeType.LINE_THICK.value: {"color": "gray", "width": 2.0, "dash": [1, 0]},
    EdgeType.LINE_THICK_DASHED.value: {"color": "gray", "width": 2.0, "dash": [5, 5]},
    EdgeType.PEDESTRIAN_MARKING.value: {"color": "orange", "width": 1.5, "dash": [5, 5]},
    EdgeType.BIKE_MARKING.value: {"color": "green", "width": 1.5, "dash": [5, 5]},
    EdgeType.GUARD_RAIL.value: {"color": "purple", "width": 2.0, "dash": [1, 0]},
    EdgeType.STOP.value: {"color": "red", "width": 3.0, "dash": [1, 0]},
    EdgeType.LINE_THIN_DOUBLE.value: {"color": "#D4AF37", "width": 2.0, "dash": [1, 0]},
    EdgeType.LINE_THIN_DOUBLE_DASHED.value: {"color": "#D4AF37", "width": 2.0, "dash": [5, 5]},
}

_ALL_EDGE_TYPES: Final[dict[int, str]] = {
    item.value: item.name.replace("_", " ").title() for item in EdgeType
}


@lru_cache(maxsize=1)
def _get_altair_style_dicts() -> tuple[dict[str, str], dict[str, float], dict[str, list[int]]]:
    """Cache the styling mappings as dictionaries keyed by the string name."""
    colors, widths, dashes = {}, {}, {}
    for val, name in _ALL_EDGE_TYPES.items():
        style = _STYLE_MAP.get(val, {"color": "gray", "width": 1.0, "dash": [1, 0]})
        colors[name] = style["color"]
        widths[name] = style["width"]
        dashes[name] = style["dash"]
    return colors, widths, dashes
