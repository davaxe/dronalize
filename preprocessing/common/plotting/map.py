from __future__ import annotations

from typing import TYPE_CHECKING, Final

from preprocessing.core._compat import require_optional
from preprocessing.core.datatypes.categories import EdgeType

if TYPE_CHECKING:
    import altair as alt

    from preprocessing.core.datatypes.map_graph import MapGraph


def plot_map_graph(
    graph: MapGraph,
    width: int = 800,
    height: int = 800,
    alpha: float = 0.7,
    *,
    include_nodes: bool = False,
) -> alt.Chart:
    """Plot a MapGraph using Altair.

    Args:
        graph: The MapGraph to plot.
        width: Width of the chart in pixels. Defaults to 800.
        height: Height of the chart in pixels. Defaults to 800.
        alpha: Transparency level for the edges. Defaults to 0.7.
        include_nodes: Whether to include node markers in the plot. Defaults to False.

    Returns:
        An Altair Chart object.

    """
    alt = require_optional("altair", extra="plot")
    pd = require_optional("pandas", extra="plot")

    if graph.num_edges == 0:
        return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("Empty Graph"))

    # 1. Prepare segments into a DataFrame
    start_indices = graph.edge_indices[0]
    end_indices = graph.edge_indices[1]

    start_pos = graph.node_positions[start_indices]
    end_pos = graph.node_positions[end_indices]

    df_edges = pd.DataFrame({
        "x1": start_pos[:, 0],
        "y1": start_pos[:, 1],
        "x2": end_pos[:, 0],
        "y2": end_pos[:, 1],
        "edge_type_int": graph.edge_types,
    })

    # Map integer types to human-readable names
    df_edges["edge_type"] = df_edges["edge_type_int"].apply(
        lambda x: (
            EdgeType(x).name.replace("_", " ").title()
            if x in {item.value for item in EdgeType}
            else EdgeType.VIRTUAL.name.title()
        )
    )

    # Extract unique types present in the data to build scales
    present_types = df_edges["edge_type"].unique().tolist()

    domain = []
    range_color = []
    range_width = []
    range_dash = []

    for pt in present_types:
        style = _STYLE_MAP.get(pt, {"color": "gray", "width": 1.0, "dash": [1, 0]})
        domain.append(pt)
        range_color.append(style["color"])
        range_width.append(style["width"])
        range_dash.append(style["dash"])

    # Disable zero-baseline to mimic Matplotlib's auto-scaling logic for spatial coordinates
    axis_scale = alt.Scale(zero=False)

    # 3. Create the lines chart
    lines = (
        alt
        .Chart(df_edges)
        .mark_rule(opacity=alpha)
        .encode(
            x=alt.X("x1", scale=axis_scale, title="X"),
            y=alt.Y("y1", scale=axis_scale, title="Y"),
            x2="x2",
            y2="y2",
            color=alt.Color(
                "edge_type:N",
                scale=alt.Scale(domain=domain, range=range_color),
                title="Edge Type",
            ),
            size=alt.Size(
                "edge_type:N", scale=alt.Scale(domain=domain, range=range_width), legend=None
            ),
            strokeDash=alt.StrokeDash(
                "edge_type:N", scale=alt.Scale(domain=domain, range=range_dash), legend=None
            ),
            tooltip=[
                alt.Tooltip("edge_type:N", title="Type"),
                alt.Tooltip("x1:Q", title="Start X", format=".2f"),
                alt.Tooltip("y1:Q", title="Start Y", format=".2f"),
                alt.Tooltip("x2:Q", title="End X", format=".2f"),
                alt.Tooltip("y2:Q", title="End Y", format=".2f"),
            ],
        )
    )

    chart = lines

    # 4. Create the nodes chart if requested
    if include_nodes and graph.num_nodes > 0:
        df_nodes = pd.DataFrame({
            "x": graph.node_positions[:, 0],
            "y": graph.node_positions[:, 1],
        })
        nodes = (
            alt
            .Chart(df_nodes)
            .mark_circle(color="black", size=15)
            .encode(
                x=alt.X("x", scale=axis_scale),
                y=alt.Y("y", scale=axis_scale),
                tooltip=[
                    alt.Tooltip("x:Q", title="Node X", format=".2f"),
                    alt.Tooltip("y:Q", title="Node Y", format=".2f"),
                ],
            )
        )
        chart = lines + nodes

    # 5. Apply chart properties and interactivity
    return chart.properties(width=width, height=height).interactive()


_STYLE_MAP: Final[dict] = {
    EdgeType.NONE.name.title(): {"color": "gray", "width": 0.5, "dash": [1, 0]},
    EdgeType.ROAD_BORDER.name.replace("_", " ").title(): {
        "color": "black",
        "width": 2.0,
        "dash": [1, 0],
    },
    EdgeType.CURB.name.title(): {"color": "black", "width": 2.5, "dash": [1, 0]},
    EdgeType.REGULATORY.name.title(): {"color": "red", "width": 1.0, "dash": [1, 0]},
    EdgeType.VIRTUAL.name.title(): {"color": "blue", "width": 0.5, "dash": [2, 2]},
    EdgeType.LINE_THIN.name.replace("_", " ").title(): {
        "color": "gray",
        "width": 1.0,
        "dash": [1, 0],
    },
    EdgeType.LINE_THIN_DASHED.name.replace("_", " ").title(): {
        "color": "gray",
        "width": 1.0,
        "dash": [5, 5],
    },
    EdgeType.LINE_THICK.name.replace("_", " ").title(): {
        "color": "gray",
        "width": 2.0,
        "dash": [1, 0],
    },
    EdgeType.LINE_THICK_DASHED.name.replace("_", " ").title(): {
        "color": "gray",
        "width": 2.0,
        "dash": [5, 5],
    },
    EdgeType.PEDESTRIAN_MARKING.name.replace("_", " ").title(): {
        "color": "orange",
        "width": 1.5,
        "dash": [5, 5],
    },
    EdgeType.BIKE_MARKING.name.replace("_", " ").title(): {
        "color": "green",
        "width": 1.5,
        "dash": [5, 5],
    },
    EdgeType.GUARD_RAIL.name.replace("_", " ").title(): {
        "color": "purple",
        "width": 2.0,
        "dash": [1, 0],
    },
    EdgeType.STOP.name.title(): {"color": "red", "width": 3.0, "dash": [1, 0]},
    EdgeType.LINE_THIN_DOUBLE.name.replace("_", " ").title(): {
        "color": "#D4AF37",
        "width": 2.0,
        "dash": [1, 0],
    },  # Gold/Yellow
    EdgeType.LINE_THIN_DOUBLE_DASHED.name.replace("_", " ").title(): {
        "color": "#D4AF37",
        "width": 2.0,
        "dash": [5, 5],
    },
}
