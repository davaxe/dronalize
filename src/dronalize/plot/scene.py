"""Altair helpers for visualizing scenes and scene records."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import polars as pl

from dronalize.core.categories import AgentCategory, EdgeType
from dronalize.core.maps import MapGraph
from dronalize.core.optional import raise_missing_optional_dependency
from dronalize.core.scene import Scene
from dronalize.plot.theme import LIGHT_THEME, EdgeStyle, PlotTheme, ThemeName, get_plot_theme

try:
    import altair as alt
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="Scene plotting", extra="plot")

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.io import SceneRecord

AspectMode = Literal["auto", "equal"]
"""Aspect-ratio modes supported by [`plot_scene`][dronalize.plot.plot_scene].

`"auto"` uses the default plotting dimensions unless explicit dimensions are
supplied, while `"equal"` computes plot bounds so one data unit in x matches
one data unit in y.
"""

_DEFAULT_WIDTH: Final[int] = 800
_DEFAULT_HEIGHT: Final[int] = 450
_MIN_DIMENSION: Final[int] = 240
_MAX_DIMENSION: Final[int] = 1200
_MAX_EQUAL_RATIO: Final[float] = 6.0
_MAX_TOOLTIP_AGENTS: Final[int] = 150
_SCREENING_PASSED: Final[str] = "Passed"
_SCREENING_FAILED: Final[str] = "Failed"
_FILTER_ALL: Final[str] = "All"
_PHASE_HISTORY: Final[str] = "History"
_PHASE_FUTURE: Final[str] = "Future"


@dataclass(slots=True, frozen=True)
class PlotSceneData:
    """Normalized plotting payload shared across scene-like inputs."""

    scene_number: int
    trajectories: pl.DataFrame
    map_edges: pl.DataFrame | None


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
    save_path: Path | str | None = None,
    ignore_map_types: set[EdgeType] | None = None,
    ignore_agent_categories: set[AgentCategory] | None = None,
    theme: ThemeName | PlotTheme = "light",
) -> alt.Chart | alt.LayerChart:
    """Plot one scene or scene record using Altair.

    Parameters
    ----------
    data : Scene or SceneRecord
        The scene or scene record to plot.
    show_map : bool, optional
        Whether to include the map graph in the plot regardless of map data
        presence. Defaults to `True`.
    aspect : AspectMode, optional
        Aspect ratio mode for the plot. `"auto"` (default) automatically adjusts
        the aspect ratio based on data bounds, while `"equal"` forces a 1:1
        aspect ratio regardless of data. Ignored if both width and height are
        specified.
    width : int, optional
        Width of the plot in pixels. If `None`, width is determined by aspect
        mode.
    height : int, optional
        Height of the plot in pixels. If `None`, height is determined by aspect
        mode.
    max_agents : int, optional
        Maximum number of unique agents to plot. If the scene contains more than
        this number, a random sample of agents will be plotted. If `None`
        (default) all agents will be plotted.
    agent_sample_seed : int, optional
        Random seed for reproducible agent sampling when `max_agents` is set.
    include_map_nodes : bool, optional
        Whether to include node markers when plotting the map graph. Defaults to
        `False`.
    save_path : Path or str, optional
        Path to save the plot (json, html, png, etc.). If `None` (default), the
        plot is not saved to disk.
    ignore_map_types : set[EdgeType], optional
        Map edge types to exclude from plotting. If `None` (default), all map
        edge types are considered.
    ignore_agent_categories : set[AgentCategory], optional
        Agent categories to exclude from plotting. If `None` (default), all
        agent categories are considered.
    theme : {"light", "dark"} or PlotTheme, optional
        Plot theme to apply. Built-in themes are `"light"` (default) and
        `"dark"`, but a custom `PlotTheme` instance can also be supplied.

    Returns
    -------
    alt.Chart or alt.LayerChart
        An Altair chart object representing the plotted scene, that can be
        further customized or saved by the caller.
    """
    _ = alt.data_transformers.enable("vegafusion")  # pyright: ignore[reportUnknownVariableType]

    normalized = _normalize_plot_input(
        data,
        max_agents=max_agents,
        agent_sample_seed=agent_sample_seed,
        ignore_map_types=ignore_map_types,
        ignore_agent_categories=ignore_agent_categories,
    )
    resolved_theme = get_plot_theme(theme)
    chart = _build_scene_chart(
        normalized,
        show_map=show_map,
        aspect=aspect,
        width=width,
        height=height,
        include_map_nodes=include_map_nodes,
        theme=resolved_theme,
    )
    if save_path is not None:
        chart.save(save_path)  # pyright: ignore[reportUnknownMemberType]
    return chart


def _build_scene_chart(
    data: PlotSceneData,
    *,
    show_map: bool,
    aspect: AspectMode,
    width: int | None,
    height: int | None,
    include_map_nodes: bool,
    theme: PlotTheme,
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

    if show_map and data.map_edges is not None and not data.map_edges.is_empty():
        map_layers = _build_map_layers(
            data.map_edges,
            include_nodes=include_map_nodes,
            theme=theme,
        )
        layers.extend(map_layers)

    if not data.trajectories.is_empty():
        trajectory_layers, trajectory_params = _build_trajectory_layers(
            data.trajectories,
            theme=theme,
        )
        layers.extend(trajectory_layers)
        params.extend(trajectory_params)

    if not layers:
        return _empty_chart("Empty Scene Plot", theme=theme)

    chart = alt.layer(*layers)
    if params:
        chart = chart.add_params(*params)

    return _style_scene_chart(
        chart.resolve_scale(
            color="independent",
            size="independent",
            strokeDash="independent",
        ),
        data=data,
        plot_width=plot_width,
        plot_height=plot_height,
        theme=theme,
    ).interactive()


def _normalize_plot_input(
    data: Scene | SceneRecord,
    *,
    max_agents: int | None,
    agent_sample_seed: int | None,
    ignore_map_types: set[EdgeType] | None,
    ignore_agent_categories: set[AgentCategory] | None,
) -> PlotSceneData:
    """Normalize a scene-like input to the plotting payload."""
    if isinstance(data, Scene):
        return _normalize_scene(
            data,
            max_agents=max_agents,
            agent_sample_seed=agent_sample_seed,
            ignore_map_types=ignore_map_types,
            ignore_agent_categories=ignore_agent_categories,
        )
    return _normalize_scene_record(
        data,
        max_agents=max_agents,
        agent_sample_seed=agent_sample_seed,
        ignore_map_types=ignore_map_types,
        ignore_agent_categories=ignore_agent_categories,
    )


def _normalize_scene(
    scene: Scene,
    *,
    max_agents: int | None,
    agent_sample_seed: int | None,
    ignore_map_types: set[EdgeType] | None,
    ignore_agent_categories: set[AgentCategory] | None,
) -> PlotSceneData:
    """Convert a `Scene` into the normalized plotting payload."""
    trajectories = scene.frame.select(
        pl.col("frame").cast(pl.Int64()),
        pl.col("id").cast(pl.Int64()).alias("agent"),
        pl.col("x").cast(pl.Float64()),
        pl.col("y").cast(pl.Float64()),
        pl.col("agent_category").cast(pl.Int64()),
    )

    if ignore_agent_categories:
        ignored_agent_values = [item.value for item in ignore_agent_categories]
        trajectories = trajectories.filter(~pl.col("agent_category").is_in(ignored_agent_values))

    trajectories = trajectories.with_columns(
        _scene_screening_status_expr(scene.passed_agent_ids),
        pl
        .col("agent_category")
        .replace_strict(_ALL_AGENT_CATEGORIES, default="Unknown")
        .alias("agent_type"),
    ).drop("agent_category")
    trajectories = _add_temporal_phase(trajectories, history_frames=scene.history_frames)
    trajectories = _sample_agents(
        trajectories, max_agents=max_agents, agent_sample_seed=agent_sample_seed
    )
    trajectories = _enrich_trajectories(trajectories)

    graph = scene.resolve_map()
    map_edges = _map_graph_to_edge_frame(graph, ignore_map_types=ignore_map_types)

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
    ignore_map_types: set[EdgeType] | None,
    ignore_agent_categories: set[AgentCategory] | None,
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

    ignored_agent_values: set[int] = (
        {item.value for item in ignore_agent_categories}
        if ignore_agent_categories is not None
        else set()
    )

    rows: list[dict[str, float | int | str]] = []
    num_agents, _num_frames = mask.shape
    for agent_idx in range(num_agents):
        agent_type_value = int(record.agent_types[agent_idx])
        if agent_type_value in ignored_agent_values:
            continue

        valid_frames = np.flatnonzero(mask[agent_idx])
        agent_type = _agent_category_name(agent_type_value)
        rows.extend(
            {
                "frame": frame_idx,
                "agent": agent_idx,
                "x": float(features[agent_idx, frame_idx, 0]) + float(offset[0]),
                "y": float(features[agent_idx, frame_idx, 1]) + float(offset[1]),
                "agent_type": agent_type,
                "screening_status": _screening_status_name(
                    passed_screening=record.passed_agent_mask[agent_idx]
                ),
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
            "screening_status": pl.String(),
        },
        orient="row",
    )
    trajectories = _add_temporal_phase(trajectories, history_frames=record.input_mask.shape[1])
    trajectories = _sample_agents(
        trajectories, max_agents=max_agents, agent_sample_seed=agent_sample_seed
    )
    trajectories = _enrich_trajectories(trajectories)

    graph = MapGraph(
        node_positions=np.asarray(record.map_node_positions, dtype=np.float64) + offset,
        edge_indices=np.asarray(record.map_edge_indices, dtype=np.int32),
        node_types=np.asarray(record.map_node_types, dtype=np.int32),
        edge_types=np.asarray(record.map_edge_types, dtype=np.int32),
    )

    return PlotSceneData(
        scene_number=record.scene_number,
        trajectories=trajectories,
        map_edges=_map_graph_to_edge_frame(graph, ignore_map_types=ignore_map_types),
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


def _enrich_trajectories(trajectories: pl.DataFrame) -> pl.DataFrame:
    """Add segment ids and agent-level frame stats used by trajectory tooltips."""
    if trajectories.is_empty():
        return trajectories.with_columns(
            pl.lit(0).cast(pl.Int64()).alias("segment"),
            pl.lit(0).cast(pl.Int64()).alias("total_frames"),
            pl.lit(0).cast(pl.Int64()).alias("missing_frames"),
        )

    trajectories = trajectories.sort(["agent", "frame"])
    frame_delta = (pl.col("frame") - pl.col("frame").shift(1).over("agent")).fill_null(1)
    gap_size = (frame_delta - 1).clip(lower_bound=0)
    return trajectories.with_columns(
        frame_delta.gt(1).cast(pl.Int64()).cum_sum().over("agent").alias("segment"),
        pl.len().over("agent").cast(pl.Int64()).alias("total_frames"),
        gap_size.sum().over("agent").cast(pl.Int64()).alias("missing_frames"),
    )


def _add_temporal_phase(trajectories: pl.DataFrame, *, history_frames: int) -> pl.DataFrame:
    """Label each plotted frame as history or future using scene chronology."""
    if trajectories.is_empty():
        return trajectories.with_columns(pl.lit(_PHASE_HISTORY).cast(pl.String()).alias("phase"))

    phase_by_frame = (
        trajectories
        .select("frame")
        .unique()
        .sort("frame")
        .with_row_index("frame_order")
        .with_columns(
            pl
            .when(pl.col("frame_order") < history_frames)
            .then(pl.lit(_PHASE_HISTORY))
            .otherwise(pl.lit(_PHASE_FUTURE))
            .alias("phase")
        )
        .select("frame", "phase")
    )
    return trajectories.join(phase_by_frame, on="frame", how="left")


def _map_graph_to_edge_frame(
    graph: MapGraph | None,
    *,
    ignore_map_types: set[EdgeType] | None = None,
) -> pl.DataFrame | None:
    """Convert a map graph to the minimal edge dataframe needed by Altair."""
    if graph is None or graph.num_edges == 0:
        return None

    edge_mask = np.ones(graph.num_edges, dtype=bool)
    if ignore_map_types:
        ignored_type_values = [item.value for item in ignore_map_types]
        edge_mask = ~np.isin(graph.edge_types, ignored_type_values)
        if not np.any(edge_mask):
            return None

    start_indices = graph.edge_indices[0][edge_mask]
    end_indices = graph.edge_indices[1][edge_mask]
    start_pos = graph.node_positions[start_indices]
    end_pos = graph.node_positions[end_indices]
    edge_types = graph.edge_types[edge_mask]

    return (
        pl
        .DataFrame({
            "x1": start_pos[:, 0],
            "y1": start_pos[:, 1],
            "x2": end_pos[:, 0],
            "y2": end_pos[:, 1],
            "edge_type_int": edge_types,
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


def _resolve_plot_dimensions(
    *,
    trajectories: pl.DataFrame,
    map_edges: pl.DataFrame | None,
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
    map_edges: pl.DataFrame | None,
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

    if show_map and map_edges is not None and not map_edges.is_empty():
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
    trajectories: pl.DataFrame,
    *,
    theme: PlotTheme = LIGHT_THEME,
) -> tuple[list[alt.Chart], list[Any]]:
    """Build Altair layers for scene trajectories."""
    unique_agents = trajectories["agent"].n_unique()
    agent_selection = alt.selection_point(fields=["agent"], on="click", clear="dblclick")
    agent_focus = alt.selection_point(
        fields=["agent"],
        on="click",
        clear="dblclick",
        empty=False,
    )
    current_frame = _frame_param(trajectories)
    category_filter, category_expr = _filter_param(
        name="agent_type_filter",
        label="Agent Type",
        field="agent_type",
        options=trajectories["agent_type"].unique().sort().to_list(),
    )
    screening_filter, screening_expr = _filter_param(
        name="screening_filter",
        label="Screening",
        field="screening_status",
        options=_screening_statuses(trajectories),
    )
    visible_frame_expr = _frame_visible_expr(current_frame)
    agent_color = _agent_color_encoding(trajectories, theme=theme)
    filters = (category_expr, screening_expr, visible_frame_expr)

    line_tooltips = _trajectory_tooltips(include_point_details=unique_agents <= _MAX_TOOLTIP_AGENTS)
    marker_tooltips = _trajectory_tooltips(include_point_details=True)

    base = _filtered_chart(trajectories, *filters).encode(
        x=alt.X("x:Q", title="X", scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title="Y", scale=alt.Scale(zero=False)),
        color=agent_color,
        order=alt.Order("frame:Q"),
        detail="segment:N",
        tooltip=line_tooltips,
    )

    trajectory_halo = base.mark_line(
        color=theme.trajectory_halo_color,
        strokeWidth=4.8,
        strokeCap="round",
        strokeJoin="round",
    ).encode(
        opacity=alt.when(agent_selection).then(alt.value(0.92)).otherwise(alt.value(0.1)),
    )

    trajectories_layer = base.encode(
        opacity=alt.when(agent_selection).then(alt.value(0.86)).otherwise(alt.value(0.14))
    ).mark_line(
        strokeWidth=1.9,
        strokeCap="round",
        strokeJoin="round",
    )

    focused_halo = (
        base
        .transform_filter(agent_focus)
        .mark_line(
            color=theme.trajectory_halo_color,
            strokeWidth=5.0,
            strokeCap="round",
            strokeJoin="round",
        )
        .encode(opacity=alt.value(0.95))
    )

    focused_trajectory = (
        base
        .transform_filter(agent_focus)
        .mark_line(
            strokeWidth=2.55,
            strokeCap="round",
            strokeJoin="round",
        )
        .encode(opacity=alt.value(1.0))
    )

    start_df = trajectories.group_by("agent", maintain_order=True).head(1)

    start_markers = (
        _filtered_chart(start_df, *filters)
        .mark_point(
            shape="circle",
            size=90,
            filled=True,
            fill="#2ecc71",
            fillOpacity=0.95,
            stroke=theme.trajectory_halo_color,
            strokeWidth=2,
        )
        .encode(x="x:Q", y="y:Q", tooltip=marker_tooltips)
    )

    current_points = (
        _filtered_chart(trajectories, category_expr, screening_expr)
        .transform_filter(_frame_current_expr(current_frame))
        .mark_point(
            shape="diamond",
            size=100,
            filled=True,
            fillOpacity=0.98,
            stroke=theme.trajectory_halo_color,
            strokeWidth=1.8,
        )
        .encode(
            x="x:Q",
            y="y:Q",
            color=agent_color,
            tooltip=marker_tooltips,
            opacity=alt.value(0.94),
        )
    )

    hovered_data_points = (
        _filtered_chart(trajectories, *filters)
        .transform_filter(agent_focus)
        .mark_point(
            size=42,
            filled=True,
            fillOpacity=0.98,
            stroke=theme.trajectory_halo_color,
            strokeWidth=1.15,
        )
        .encode(
            x="x:Q",
            y="y:Q",
            color=agent_color,
            tooltip=marker_tooltips,
        )
    )

    selection_targets = base.mark_line(
        color="#000000",
        strokeWidth=14,
        strokeCap="round",
        strokeJoin="round",
        opacity=0.001,
    ).add_params(agent_selection, agent_focus)

    return [
        selection_targets,
        trajectory_halo,
        trajectories_layer,
        focused_halo,
        focused_trajectory,
        start_markers,
        current_points,
        hovered_data_points,
    ], [current_frame, category_filter, screening_filter]


def _trajectory_tooltips(
    *,
    include_point_details: bool,
) -> list[alt.Tooltip]:
    """Build a consistent tooltip payload for trajectory layers."""
    tooltips: list[alt.Tooltip] = [
        alt.Tooltip("agent:Q", title="Agent"),
        alt.Tooltip("agent_type:N", title="Type"),
        alt.Tooltip("phase:N", title="Phase"),
        alt.Tooltip("screening_status:N", title="Screening"),
        alt.Tooltip("total_frames:Q", title="Total Frames"),
        alt.Tooltip("missing_frames:Q", title="Missing Frames"),
    ]
    if include_point_details:
        tooltips.extend([
            alt.Tooltip("frame:Q", title="Frame"),
            alt.Tooltip("x:Q", format=".2f"),
            alt.Tooltip("y:Q", format=".2f"),
        ])
    return tooltips


def _build_map_layers(
    map_edges: pl.DataFrame,
    *,
    include_nodes: bool,
    theme: PlotTheme = LIGHT_THEME,
) -> list[alt.Chart]:
    """Build Altair layers for a normalized edge dataframe."""
    present_types = map_edges["edge_type"].unique().to_list()
    present_types.sort()
    edge_styles = _edge_styles_by_name(theme.name)

    axis_scale = alt.Scale(zero=False)
    plot_color = [
        edge_styles.get(edge_type, _DEFAULT_EDGE_STYLE).color for edge_type in present_types
    ]
    plot_width = [
        edge_styles.get(edge_type, _DEFAULT_EDGE_STYLE).width for edge_type in present_types
    ]
    plot_dash = [
        edge_styles.get(edge_type, _DEFAULT_EDGE_STYLE).dash for edge_type in present_types
    ]

    lines = (
        alt
        .Chart(map_edges)
        .mark_rule(strokeCap="round")
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
            opacity=alt.value(0.34),
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
            .mark_circle(color=theme.map_node_color, size=18, opacity=0.35)
            .encode(
                x=alt.X("x1:Q", scale=axis_scale),
                y=alt.Y("y1:Q", scale=axis_scale),
                opacity=alt.value(0.28),
            )
        )
        end_nodes = (
            alt
            .Chart(map_edges)
            .mark_circle(color=theme.map_node_color, size=18, opacity=0.35)
            .encode(
                x=alt.X("x2:Q", scale=axis_scale),
                y=alt.Y("y2:Q", scale=axis_scale),
                opacity=alt.value(0.28),
            )
        )
        layers.extend((start_nodes, end_nodes))

    return layers


def _empty_chart(message: str, *, theme: PlotTheme = LIGHT_THEME) -> alt.Chart:
    """Create a simple text chart used for empty plotting inputs."""
    return (
        alt
        .Chart(pl.DataFrame())
        .mark_text(size=16, color=theme.title_color, fontWeight="bold")
        .encode(text=alt.value(message))
    )


def _filtered_chart(data: pl.DataFrame, *filters: Any) -> alt.Chart:
    """Return an Altair chart with all provided filters applied in order."""
    chart = alt.Chart(data)
    for filter_ in filters:
        chart = chart.transform_filter(filter_)
    return chart


_DEFAULT_EDGE_STYLE: Final[EdgeStyle] = EdgeStyle(color="gray", width=1.0, dash=(1, 0))

_ALL_EDGE_TYPES: Final[dict[int, str]] = {
    item.value: item.name.replace("_", " ").title() for item in EdgeType
}
_ALL_AGENT_CATEGORIES: Final[dict[int, str]] = {
    item.value: item.name.replace("_", " ").title() for item in AgentCategory
}


@lru_cache(maxsize=2)
def _edge_styles_by_name(theme_name: str) -> dict[str, EdgeStyle]:
    """Return one theme's edge styles keyed by display name."""
    theme = get_plot_theme(theme_name)
    return {_ALL_EDGE_TYPES[value]: style for value, style in theme.edge_styles.items()}


def _agent_category_name(value: int) -> str:
    """Return the display name for one encoded agent category."""
    return _ALL_AGENT_CATEGORIES.get(value, "Unknown")


def _agent_color_encoding(
    trajectories: pl.DataFrame,
    *,
    theme: PlotTheme = LIGHT_THEME,
) -> alt.Color:
    """Return a stable per-agent color encoding that does not change under filtering."""
    agent_ids = trajectories["agent"].unique().sort().to_list()
    return alt.Color(
        "agent:N",
        legend=None,
        scale=alt.Scale(domain=agent_ids, scheme=theme.agent_color_scheme),
    )


def _style_scene_chart(
    chart: alt.Chart | alt.LayerChart,
    *,
    data: PlotSceneData,
    plot_width: int,
    plot_height: int,
    theme: PlotTheme = LIGHT_THEME,
) -> alt.Chart | alt.LayerChart:
    """Apply the shared theme used by scene plots."""
    return (
        chart
        .properties(
            width=plot_width,
            height=plot_height,
            title=_scene_title(data),
        )
        .configure(background=theme.chart_background)
        .configure_view(fill=theme.plot_background, stroke=theme.grid_color, strokeOpacity=0.65)
        .configure_title(color=theme.title_color, fontSize=16, fontWeight="bold", offset=12)
        .configure_axis(
            grid=True,
            gridColor=theme.grid_color,
            gridOpacity=0.35,
            gridDash=[2, 3],
            domainColor=theme.grid_color,
            tickColor=theme.grid_color,
            labelColor=theme.axis_color,
            titleColor=theme.title_color,
        )
        .configure_legend(
            titleColor=theme.title_color,
            labelColor=theme.axis_color,
            fillColor=theme.plot_background,
            strokeColor=theme.grid_color,
            padding=10,
            cornerRadius=8,
            orient="bottom",
        )
    )


def _scene_screening_status_expr(passed_agent_ids: frozenset[int] | None) -> pl.Expr:
    """Return a per-agent screening-status expression for scene dataframes."""
    if passed_agent_ids is None:
        return pl.lit(_SCREENING_PASSED).alias("screening_status")
    return (
        pl
        .when(pl.col("agent").is_in(list(passed_agent_ids)))
        .then(pl.lit(_SCREENING_PASSED))
        .otherwise(pl.lit(_SCREENING_FAILED))
        .alias("screening_status")
    )


def _screening_status_name(*, passed_screening: bool | np.bool_) -> str:
    """Return the display label for one screening outcome."""
    return _SCREENING_PASSED if bool(passed_screening) else _SCREENING_FAILED


def _screening_statuses(trajectories: pl.DataFrame) -> list[str]:
    """Return screening statuses present in plotting order."""
    present = set(trajectories["screening_status"].unique().to_list())
    return [status for status in (_SCREENING_PASSED, _SCREENING_FAILED) if status in present]


def _frame_param(trajectories: pl.DataFrame) -> Any:
    """Return the current-frame slider parameter."""
    frame_min = int(trajectories["frame"].min())
    frame_max = int(trajectories["frame"].max())
    return alt.param(
        name="selected_frame",
        value=frame_max,
        bind=alt.binding_range(min=frame_min, max=frame_max, step=1, name="Frame "),
    )


def _frame_visible_expr(current_frame: Any) -> str:
    """Return the filter expression for rows visible at the selected frame."""
    return f"datum.frame <= {current_frame.name}"


def _frame_current_expr(current_frame: Any) -> str:
    """Return the filter expression for rows exactly on the selected frame."""
    return f"datum.frame == {current_frame.name}"


def _filter_param(
    *,
    name: str,
    label: str,
    field: str,
    options: list[str],
) -> tuple[Any, str]:
    """Create a dropdown filter parameter and its Vega filter expression."""
    param = alt.param(
        name=name,
        value=_FILTER_ALL,
        bind=alt.binding_select(options=[_FILTER_ALL, *options], name=f"{label} "),
    )
    return param, f"{name} == '{_FILTER_ALL}' || datum.{field} == {name}"


def _scene_title(data: PlotSceneData) -> alt.TitleParams:
    """Return the chart title and summary subtitle for one plotted scene."""
    agent_count = 0
    category_count = 0
    passed_count = 0
    failed_count = 0
    frame_range = "frames: n/a"
    if not data.trajectories.is_empty():
        agent_summary = data.trajectories.select("agent", "agent_type", "screening_status").unique(
            subset="agent"
        )
        agent_count = agent_summary.height
        category_count = agent_summary["agent_type"].n_unique()
        passed_count = int((agent_summary["screening_status"] == _SCREENING_PASSED).sum())
        failed_count = int((agent_summary["screening_status"] == _SCREENING_FAILED).sum())
        frame_min = int(data.trajectories["frame"].min())
        frame_max = int(data.trajectories["frame"].max())
        frame_range = f"frames: {frame_min}-{frame_max}"

    edge_count = 0 if data.map_edges is None else data.map_edges.height
    subtitle = (
        f"{agent_count} agents | {category_count} categories | "
        f"{passed_count} passed / {failed_count} failed | "
        f"{edge_count} map edges | {frame_range}"
    )
    return alt.TitleParams(text=f"Scene {data.scene_number}", subtitle=[subtitle], anchor="start")
