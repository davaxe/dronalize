# pyright: standard

"""Overlay plotting helpers that combine trajectories and map context."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import altair as alt  # Optional dependency, checked at import time (in __init__.py)

from dronalize.plot.map import _build_map_graph_layers, _empty_chart
from dronalize.plot.trajectory import _build_trajectory_layers, _sample_trajectory_groups

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import polars as pl

    from dronalize.core.maps import MapGraph

Chart = alt.Chart | alt.LayerChart


def plot_trajectories_on_map(
    data: pl.DataFrame,
    graph: MapGraph,
    x_col: str = "x",
    y_col: str = "y",
    group_by: str | None = None,
    n_groups: int | None = None,
    frame_col: str = "frame",
    width: int = 700,
    height: int = 700,
    map_alpha: float = 0.7,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    save_path: Path | str | None = None,
    title: str | None = None,
    group_sample_seed: int | None = None,
    highlight_frame: int | Sequence[int] | None = None,
    include_map_nodes: bool = False,
    disable_max_rows: bool = True,
    **kwargs: Any,  # noqa: ANN401
) -> Chart:
    """Plot trajectories and a map graph in the same Altair figure."""
    alt.renderers.enable("browser")

    if disable_max_rows:
        alt.data_transformers.disable_max_rows()

    data = _sample_trajectory_groups(
        data, group_by=group_by, n_groups=n_groups, group_sample_seed=group_sample_seed
    )

    layers: list[alt.Chart] = []
    params: list[Any] = []

    if graph.num_edges > 0:
        map_layers, edge_selection = _build_map_graph_layers(
            graph, alt=alt, alpha=map_alpha, include_nodes=include_map_nodes
        )
        layers.extend(map_layers)
        params.append(edge_selection)

    if not data.is_empty():
        layers.extend(
            _build_trajectory_layers(
                data,
                alt=alt,
                x_col=x_col,
                y_col=y_col,
                group_by=group_by,
                frame_col=frame_col,
                x_label=x_label,
                y_label=y_label,
                highlight_frame=highlight_frame,
            )
        )

    if not layers:
        chart = _empty_chart(alt, "Empty Graph And Trajectory Data")
    else:
        chart = alt.layer(*layers)
        if params:
            chart = chart.add_params(*params)
        chart = (
            chart
            .properties(
                width=width,
                height=height,
                title=title or "Trajectory and Map Graph Visualization",
                **kwargs,
            )
            .configure_view(stroke=None)
            .configure_title(fontSize=22)
            .configure_axis(labelFontSize=14, titleFontSize=16)
            .configure_legend(labelFontSize=14, titleFontSize=16)
            .interactive()
        )

    if save_path:
        chart.save(save_path)

    return chart
