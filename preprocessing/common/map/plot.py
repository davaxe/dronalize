import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from preprocessing.core.datatypes.categories import EdgeType
from preprocessing.core.datatypes.map_graph import MapGraph


def plot_map_graph(
    graph: MapGraph,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (10, 10),
    alpha: float = 0.7,
    *,
    include_nodes: bool = False,
) -> Axes:
    """Plot a MapGraph using Matplotlib.

    Args:
        graph: The MapGraph to plot.
        ax: Optional Matplotlib Axes to plot on. If None, a new figure and axes are created.
        figsize: Size of the figure if a new one is created. Defaults to (10, 10).
        alpha: Transparency level for the edges. Defaults to 0.7.
        include_nodes: Whether to include node markers in the plot. Defaults to False.

    Returns:
        The Matplotlib Axes object containing the figure.

    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    if graph.num_edges == 0:
        return ax

    # 1. Prepare segments for LineCollection
    # edge_indices shape is (2, M). Row 0 is start indices, Row 1 is end indices.
    start_indices = graph.edge_indices[0]
    end_indices = graph.edge_indices[1]

    # Gather coordinates: (M, 2)
    start_pos = graph.node_positions[start_indices]
    end_pos = graph.node_positions[end_indices]

    # Stack to get (M, 2, 2) array where each row is [[x1, y1], [x2, y2]]
    segments = np.stack((start_pos, end_pos), axis=1)

    # 2. Define Styles based on EdgeType
    # Map EdgeType to (color, linewidth, linestyle)
    style_map = {
        EdgeType.NONE: ("gray", 0.5, "solid"),
        EdgeType.ROAD_BORDER: ("black", 2.0, "solid"),
        EdgeType.CURB: ("black", 2.5, "solid"),
        EdgeType.REGULATORY: ("red", 1.0, "solid"),
        EdgeType.VIRTUAL: ("blue", 0.5, "dotted"),
        EdgeType.LINE_THIN: ("gray", 1.0, "solid"),
        EdgeType.LINE_THIN_DASHED: ("gray", 1.0, "dashed"),
        EdgeType.LINE_THICK: ("gray", 2.0, "solid"),
        EdgeType.LINE_THICK_DASHED: ("gray", 2.0, "dashed"),
        EdgeType.PEDESTRIAN_MARKING: ("orange", 1.5, "dashed"),
        EdgeType.BIKE_MARKING: ("green", 1.5, "dashed"),
        EdgeType.GUARD_RAIL: ("purple", 2.0, "solid"),
        EdgeType.STOP: ("red", 3.0, "solid"),
        EdgeType.LINE_THIN_DOUBLE: ("yellow", 2.0, "solid"),
        EdgeType.LINE_THIN_DOUBLE_DASHED: ("yellow", 2.0, "dashed"),
    }

    # defaults
    default_style = ("gray", 1.0, "solid")

    # 3. Create collections per style
    # Doing one LineCollection per style group is often cleaner than passing lists
    # of styles to a single collection, though both work. Grouping allows valid legends.

    edge_types_np = graph.edge_types

    # We iterate over unique edge types present in the graph to batch operations
    present_types = np.unique(edge_types_np)

    for et_int in present_types:
        try:
            et = EdgeType(int(et_int))
            color, lw, ls = style_map.get(et, default_style)
        except ValueError:
            color, lw, ls = default_style
            et = EdgeType.VIRTUAL

        # Boolean mask for current edge type
        mask = edge_types_np == et_int
        current_segments = segments[mask].tolist()

        lc = LineCollection(
            current_segments,
            colors=color,
            linewidths=lw,
            linestyles=ls,
            alpha=alpha,
            label=et.name.replace("_", " ").title(),
        )
        ax.add_collection(lc)

    if include_nodes and graph.num_nodes > 0:
        # Plot nodes with a simple style (could be enhanced to differentiate types)
        ax.scatter(
            graph.node_positions[:, 0],
            graph.node_positions[:, 1],
            color="black",
            s=10,
            zorder=3,
            label="Nodes",
        )

    # 4. Set plot limits and aspect
    # LineCollection does not automatically update autoscaling
    all_x = graph.node_positions[:, 0]
    all_y = graph.node_positions[:, 1]

    if len(all_x) > 0:
        ax.set_xlim(float(all_x.min()) - 5, float(all_x.max()) + 5)
        ax.set_ylim(float(all_y.min()) - 5, float(all_y.max()) + 5)

    ax.set_aspect("equal")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()

    return ax
