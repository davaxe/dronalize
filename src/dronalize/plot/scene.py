"""Altair helpers for visualizing scenes and scene records."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import polars as pl

from dronalize.core.categories import AgentCategory, EdgeType
from dronalize.core.maps import MapGraph
from dronalize.core.optional import require_optional
from dronalize.core.scene import Scene

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    import altair as alt

    from dronalize.io import SceneRecord

AspectMode = Literal["auto", "equal"]

_DEFAULT_WIDTH: Final[int] = 800
_DEFAULT_HEIGHT: Final[int] = 450
_MIN_DIMENSION: Final[int] = 240
_MAX_DIMENSION: Final[int] = 1200
_MAX_EQUAL_RATIO: Final[float] = 6.0
_MAX_TOOLTIP_AGENTS: Final[int] = 150


@dataclass(slots=True, frozen=True)
class PlotSceneData:
    """Normalized plotting payload shared across scene-like inputs."""

    scene_number: int
    trajectories: pl.DataFrame
    map_edges: pl.DataFrame


def plot_scene(
    data: Scene | SceneRecord,
    *,
    show_map: bool = True,
    aspect: AspectMode = "auto",
    width: int | None = None,
    height: int | None = None,
    max_agents: int | None = None,
    agent_sample_seed: int | None = None,
    include_map_nodes: bool = False,
    disable_max_rows: bool = True,
    save_path: Path | str | None = None,
) -> alt.Chart | alt.LayerChart:
    """Plot one scene or scene record using Altair."""
    alt = require_optional("altair", extra="plot")

    if disable_max_rows:
        alt.data_transformers.disable_max_rows()

    normalized = _normalize_plot_input(
        data,
        max_agents=max_agents,
        agent_sample_seed=agent_sample_seed,
    )

    chart = _build_scene_chart(
        normalized,
        alt=alt,
        show_map=show_map,
        aspect=aspect,
        width=width,
        height=height,
        include_map_nodes=include_map_nodes,
    )

    if save_path is not None:
        chart.save(save_path)  # pyright: ignore[reportUnknownMemberType]

    return chart


def _build_scene_chart(
    data: PlotSceneData,
    *,
    alt: ModuleType,
    show_map: bool,
    aspect: AspectMode,
    width: int | None,
    height: int | None,
    include_map_nodes: bool,
) -> alt.Chart | alt.LayerChart:
    """Construct the layered Altair chart from normalized scene data."""
    plot_width, plot_height = _resolve_plot_dimensions(
        trajectories=data.trajectories,
        map_edges=data.map_edges,
        show_map=show_map,
        aspect=aspect,
        width=width,
        height=height,
    )

    layers: list[alt.Chart] = []
    params: list[Any] = []

    if show_map and not data.map_edges.is_empty():
        map_layers, edge_selection = _build_map_layers(
            data.map_edges,
            alt=alt,
            include_nodes=include_map_nodes,
        )
        layers.extend(map_layers)
        params.append(edge_selection)

    if not data.trajectories.is_empty():
        trajectory_layers, trajectory_params = _build_trajectory_layers(
            data.trajectories,
            alt=alt,
        )
        layers.extend(trajectory_layers)
        params.extend(trajectory_params)

    if not layers:
        return _empty_chart(alt, "Empty Scene Plot")

    chart = alt.layer(*layers)
    if params:
        chart = chart.add_params(*params)

    return (
        chart
        .resolve_scale(color="independent", size="independent", strokeDash="independent")
        .properties(width=plot_width, height=plot_height)
        .configure_view(stroke=None)
        .configure_axis(grid=True, gridColor="grey", gridOpacity=0.2, gridDash=[2, 2])
        .interactive()
    )


def _normalize_plot_input(
    data: Scene | SceneRecord,
    *,
    max_agents: int | None,
    agent_sample_seed: int | None,
) -> PlotSceneData:
    """Normalize a scene-like input to the plotting payload."""
    if isinstance(data, Scene):
        return _normalize_scene(
            data,
            max_agents=max_agents,
            agent_sample_seed=agent_sample_seed,
        )
    return _normalize_scene_record(
        data,
        max_agents=max_agents,
        agent_sample_seed=agent_sample_seed,
    )


def _normalize_scene(
    scene: Scene,
    *,
    max_agents: int | None,
    agent_sample_seed: int | None,
) -> PlotSceneData:
    """Convert a `Scene` into the normalized plotting payload."""
    trajectories = scene.frame.select(
        pl.col("frame").cast(pl.Int64()),
        pl.col("id").cast(pl.Int64()).alias("agent"),
        pl.col("x").cast(pl.Float64()),
        pl.col("y").cast(pl.Float64()),
        pl.col("agent_category").cast(pl.Int64()),
    )
    trajectories = trajectories.with_columns(
        pl
        .col("agent_category")
        .replace_strict(_ALL_AGENT_CATEGORIES, default="Unknown")
        .alias("agent_type")
    ).drop("agent_category")
    trajectories = _sample_agents(
        trajectories,
        max_agents=max_agents,
        agent_sample_seed=agent_sample_seed,
    )
    trajectories = _add_segments(trajectories)

    graph = scene.resolve_map()
    map_edges = _map_graph_to_edge_frame(graph)

    return PlotSceneData(
        scene_number=scene.scene_number,
        trajectories=trajectories,
        map_edges=map_edges,
    )


def _normalize_scene_record(
    record: SceneRecord,
    *,
    max_agents: int | None,
    agent_sample_seed: int | None,
) -> PlotSceneData:
    """Convert a `SceneRecord` into the normalized plotting payload."""
    features = np.concatenate((record.input_features, record.output_features), axis=1)
    mask = np.concatenate((record.input_mask, record.output_mask), axis=1)

    if features.shape[-1] < 2:
        msg = "SceneRecord plotting requires x/y feature columns in positions 0 and 1."
        raise ValueError(msg)

    offset = np.asarray(record.position_offset, dtype=np.float64).reshape(-1)
    if offset.shape != (2,):
        msg = f"SceneRecord position_offset must resolve to shape (2,), got {offset.shape!r}."
        raise ValueError(msg)

    rows: list[dict[str, float | int | str]] = []
    num_agents, _num_frames = mask.shape
    for agent_idx in range(num_agents):
        valid_frames = np.flatnonzero(mask[agent_idx])
        agent_type = _agent_category_name(int(record.agent_types[agent_idx]))
        rows.extend(
            {
                "frame": frame_idx,
                "agent": agent_idx,
                "x": float(features[agent_idx, frame_idx, 0]) + float(offset[0]),
                "y": float(features[agent_idx, frame_idx, 1]) + float(offset[1]),
                "agent_type": agent_type,
            }
            for frame_idx in valid_frames.tolist()
        )

    trajectories = pl.DataFrame(
        rows,
        schema={
            "frame": pl.Int64(),
            "agent": pl.Int64(),
            "x": pl.Float64(),
            "y": pl.Float64(),
            "agent_type": pl.String(),
        },
        orient="row",
    )
    trajectories = _sample_agents(
        trajectories,
        max_agents=max_agents,
        agent_sample_seed=agent_sample_seed,
    )
    trajectories = _add_segments(trajectories)

    graph = MapGraph(
        node_positions=np.asarray(record.map_node_positions, dtype=np.float64) + offset,
        edge_indices=np.asarray(record.map_edge_indices, dtype=np.int32),
        node_types=np.asarray(record.map_node_types, dtype=np.int32),
        edge_types=np.asarray(record.map_edge_types, dtype=np.int32),
    )

    return PlotSceneData(
        scene_number=record.scene_number,
        trajectories=trajectories,
        map_edges=_map_graph_to_edge_frame(graph),
    )


def _sample_agents(
    trajectories: pl.DataFrame,
    *,
    max_agents: int | None,
    agent_sample_seed: int | None,
) -> pl.DataFrame:
    """Optionally subsample plotted agents."""
    if max_agents is None or trajectories.is_empty():
        return trajectories

    unique_agents = trajectories.select("agent").unique()
    if unique_agents.height <= max_agents:
        return trajectories

    selected_agents = unique_agents.sample(max_agents, seed=agent_sample_seed)
    return trajectories.join(selected_agents, on="agent", how="semi")


def _add_segments(trajectories: pl.DataFrame) -> pl.DataFrame:
    """Add per-agent segment ids so gaps are rendered as broken lines."""
    if trajectories.is_empty():
        return trajectories.with_columns(pl.lit(0).cast(pl.Int64()).alias("segment"))

    trajectories = trajectories.sort(["agent", "frame"])
    gap = (pl.col("frame") - pl.col("frame").shift(1).over("agent")).fill_null(1) > 1
    segment = gap.cast(pl.Int64()).cum_sum().over("agent")
    return trajectories.with_columns(segment.cast(pl.Int64()).alias("segment"))


def _map_graph_to_edge_frame(graph: MapGraph | None) -> pl.DataFrame:
    """Convert a map graph to the minimal edge dataframe needed by Altair."""
    if graph is None or graph.num_edges == 0:
        return _empty_map_edge_frame()

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


def _empty_map_edge_frame() -> pl.DataFrame:
    """Return the canonical empty edge dataframe."""
    return pl.DataFrame(
        schema={
            "x1": pl.Float64(),
            "y1": pl.Float64(),
            "x2": pl.Float64(),
            "y2": pl.Float64(),
            "edge_type": pl.String(),
            "length": pl.Float64(),
        }
    )


def _resolve_plot_dimensions(
    *,
    trajectories: pl.DataFrame,
    map_edges: pl.DataFrame,
    show_map: bool,
    aspect: AspectMode,
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    """Resolve chart dimensions while keeping aspect behavior explicit."""
    if width is not None and height is not None:
        return width, height
    if width is not None:
        return width, height or _DEFAULT_HEIGHT
    if height is not None:
        return width or _DEFAULT_WIDTH, height
    if aspect == "auto":
        return _DEFAULT_WIDTH, _DEFAULT_HEIGHT

    x_min, x_max, y_min, y_max = _collect_plot_bounds(
        trajectories=trajectories,
        map_edges=map_edges,
        show_map=show_map,
    )
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    ratio = min(max(x_span / y_span, 1.0 / _MAX_EQUAL_RATIO), _MAX_EQUAL_RATIO)

    if ratio >= 1.0:
        return _MAX_DIMENSION, max(_MIN_DIMENSION, round(_MAX_DIMENSION / ratio))
    return max(_MIN_DIMENSION, round(_MAX_DIMENSION * ratio)), _MAX_DIMENSION


def _collect_plot_bounds(
    *,
    trajectories: pl.DataFrame,
    map_edges: pl.DataFrame,
    show_map: bool,
) -> tuple[float, float, float, float]:
    """Return combined x/y bounds across plotted layers."""
    x_values: list[float] = []
    y_values: list[float] = []

    if not trajectories.is_empty():
        summary = trajectories.select(
            pl.col("x").min().alias("x_min"),
            pl.col("x").max().alias("x_max"),
            pl.col("y").min().alias("y_min"),
            pl.col("y").max().alias("y_max"),
        )
        x_min, x_max, y_min, y_max = summary.row(0)
        x_values.extend((float(x_min), float(x_max)))
        y_values.extend((float(y_min), float(y_max)))

    if show_map and not map_edges.is_empty():
        summary = map_edges.select(
            pl.min_horizontal(pl.col("x1"), pl.col("x2")).min().alias("x_min"),
            pl.max_horizontal(pl.col("x1"), pl.col("x2")).max().alias("x_max"),
            pl.min_horizontal(pl.col("y1"), pl.col("y2")).min().alias("y_min"),
            pl.max_horizontal(pl.col("y1"), pl.col("y2")).max().alias("y_max"),
        )
        x_min, x_max, y_min, y_max = summary.row(0)
        x_values.extend((float(x_min), float(x_max)))
        y_values.extend((float(y_min), float(y_max)))

    if not x_values or not y_values:
        return 0.0, 1.0, 0.0, 1.0

    return min(x_values), max(x_values), min(y_values), max(y_values)


def _build_trajectory_layers(
    trajectories: pl.DataFrame, *, alt: ModuleType
) -> tuple[list[alt.Chart], list[Any]]:
    """Build Altair layers for scene trajectories."""
    unique_agents = trajectories["agent"].n_unique()
    agent_selection = alt.selection_point(fields=["agent"], on="click", empty=False)
    line_tooltips: list[alt.Tooltip] = [
        alt.Tooltip("agent:Q", title="Agent"),
        alt.Tooltip("agent_type:N", title="Type"),
    ]
    marker_tooltips: list[alt.Tooltip] = [
        alt.Tooltip("agent:Q", title="Agent"),
        alt.Tooltip("agent_type:N", title="Type"),
        alt.Tooltip("frame:Q", title="Frame"),
        alt.Tooltip("x:Q", format=".2f"),
        alt.Tooltip("y:Q", format=".2f"),
    ]
    if unique_agents <= _MAX_TOOLTIP_AGENTS:
        line_tooltips.extend([
            alt.Tooltip("frame:Q", title="Frame"),
            alt.Tooltip("x:Q", format=".2f"),
            alt.Tooltip("y:Q", format=".2f"),
        ])

    base = alt.Chart(trajectories).encode(
        x=alt.X("x:Q", title="X", scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title="Y", scale=alt.Scale(zero=False)),
        color=alt.Color("agent:N", legend=None, scale=alt.Scale(scheme="category20")),
        order=alt.Order("frame:Q"),
        tooltip=line_tooltips,
    )

    trajectories_layer = base.encode(
        detail="segment:N",
        opacity=alt.when(agent_selection).then(alt.value(1.0)).otherwise(alt.value(0.8)),
    ).mark_line(strokeWidth=2)

    start_df = trajectories.group_by("agent", maintain_order=True).head(1)
    end_df = trajectories.group_by("agent", maintain_order=True).tail(1)

    start_markers = (
        alt
        .Chart(start_df)
        .mark_point(shape="circle", size=75, filled=True, fill="#2ecc71", stroke="black")
        .encode(x="x:Q", y="y:Q", tooltip=marker_tooltips)
    )
    end_markers = (
        alt
        .Chart(end_df)
        .mark_point(shape="cross", size=75, color="#e74c3c", stroke="black")
        .encode(x="x:Q", y="y:Q", tooltip=marker_tooltips)
    )

    selected_points = (
        alt
        .Chart(trajectories)
        .mark_point(size=45, filled=True, stroke="black", strokeWidth=0.5)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("agent:N", legend=None, scale=alt.Scale(scheme="category20")),
            tooltip=marker_tooltips,
        )
        .transform_filter(agent_selection)
    )

    return [trajectories_layer, start_markers, end_markers, selected_points], [agent_selection]


def _build_map_layers(
    map_edges: pl.DataFrame,
    *,
    alt: ModuleType,
    include_nodes: bool,
) -> tuple[list[alt.Chart], Any]:
    """Build Altair layers for a normalized edge dataframe."""
    color_dict, width_dict, dash_dict = _get_altair_style_dicts()
    present_types = map_edges["edge_type"].unique().to_list()
    present_types.sort()

    axis_scale = alt.Scale(zero=False)
    edge_selection = alt.selection_point(fields=["edge_type"], bind="legend")
    plot_color = [color_dict.get(edge_type, "gray") for edge_type in present_types]
    plot_width = [width_dict.get(edge_type, 1.0) for edge_type in present_types]
    plot_dash = [dash_dict.get(edge_type, [1, 0]) for edge_type in present_types]

    lines = (
        alt
        .Chart(map_edges)
        .mark_rule()
        .encode(
            x=alt.X("x1:Q", scale=axis_scale, title="X", axis=alt.Axis(grid=False)),
            y=alt.Y("y1:Q", scale=axis_scale, title="Y", axis=alt.Axis(grid=False)),
            x2="x2:Q",
            y2="y2:Q",
            color=alt.Color(
                "edge_type:N",
                scale=alt.Scale(domain=present_types, range=plot_color),
                legend=alt.Legend(
                    title="Edge Type",
                    symbolType="stroke",
                    symbolStrokeWidth=3,
                    orient="right",
                ),
            ),
            size=alt.Size(
                "edge_type:N",
                scale=alt.Scale(domain=present_types, range=plot_width),
                legend=None,
            ),
            strokeDash=alt.StrokeDash(
                "edge_type:N",
                scale=alt.Scale(domain=present_types, range=plot_dash),
                legend=None,
            ),
            opacity=alt.when(edge_selection).then(alt.value(0.7)).otherwise(alt.value(0.05)),
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
        start_nodes = (
            alt
            .Chart(map_edges)
            .mark_circle(color="black", size=15)
            .encode(
                x=alt.X("x1:Q", scale=axis_scale),
                y=alt.Y("y1:Q", scale=axis_scale),
                opacity=alt.when(edge_selection).then(alt.value(1.0)).otherwise(alt.value(0.05)),
            )
        )
        end_nodes = (
            alt
            .Chart(map_edges)
            .mark_circle(color="black", size=15)
            .encode(
                x=alt.X("x2:Q", scale=axis_scale),
                y=alt.Y("y2:Q", scale=axis_scale),
                opacity=alt.when(edge_selection).then(alt.value(1.0)).otherwise(alt.value(0.05)),
            )
        )
        layers.extend((start_nodes, end_nodes))

    return layers, edge_selection


def _empty_chart(alt: ModuleType, message: str) -> alt.Chart:
    """Create a simple text chart used for empty plotting inputs."""
    return alt.Chart(pl.DataFrame()).mark_text(size=16).encode(text=alt.value(message))


_STYLE_MAP: Final[dict[int, tuple[str, float, list[int]]]] = {
    EdgeType.NONE.value: ("gray", 0.5, [1, 0]),
    EdgeType.ROAD_BORDER.value: ("black", 2.0, [1, 0]),
    EdgeType.CURB.value: ("black", 2.5, [1, 0]),
    EdgeType.REGULATORY.value: ("red", 1.0, [1, 0]),
    EdgeType.VIRTUAL.value: ("blue", 0.5, [2, 2]),
    EdgeType.LINE_THIN.value: ("gray", 1.0, [1, 0]),
    EdgeType.LINE_THIN_DASHED.value: ("gray", 1.0, [5, 5]),
    EdgeType.LINE_THICK.value: ("gray", 2.0, [1, 0]),
    EdgeType.LINE_THICK_DASHED.value: ("gray", 2.0, [5, 5]),
    EdgeType.PEDESTRIAN_MARKING.value: ("orange", 1.5, [5, 5]),
    EdgeType.BIKE_MARKING.value: ("green", 1.5, [5, 5]),
    EdgeType.GUARD_RAIL.value: ("purple", 2.0, [1, 0]),
    EdgeType.STOP.value: ("red", 3.0, [1, 0]),
    EdgeType.LINE_THIN_DOUBLE.value: ("#D4AF37", 2.0, [1, 0]),
    EdgeType.LINE_THIN_DOUBLE_DASHED.value: ("#D4AF37", 2.0, [5, 5]),
}

_ALL_EDGE_TYPES: Final[dict[int, str]] = {
    item.value: item.name.replace("_", " ").title() for item in EdgeType
}
_ALL_AGENT_CATEGORIES: Final[dict[int, str]] = {
    item.value: item.name.replace("_", " ").title() for item in AgentCategory
}


@lru_cache(maxsize=1)
def _get_altair_style_dicts() -> tuple[dict[str, str], dict[str, float], dict[str, list[int]]]:
    """Cache Altair styling mappings keyed by edge type name."""
    colors: dict[str, str] = {}
    widths: dict[str, float] = {}
    dashes: dict[str, list[int]] = {}
    for value, name in _ALL_EDGE_TYPES.items():
        color, width, dash = _STYLE_MAP.get(value, ("gray", 1.0, [1, 0]))
        colors[name] = color
        widths[name] = width
        dashes[name] = dash
    return colors, widths, dashes


def _agent_category_name(value: int) -> str:
    """Return the display name for one encoded agent category."""
    return _ALL_AGENT_CATEGORIES.get(value, "Unknown")
