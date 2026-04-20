# pyright: standard
"""Altair helpers for visualizing scenes and scene records."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

import polars as pl

from dronalize.core.optional import raise_missing_optional_dependency
from dronalize.plot.scene_normalization import (
    EDGE_TYPE_LABELS,
    PHASE_FUTURE,
    PHASE_HISTORY,
    SCREENING_FAILED,
    SCREENING_PASSED,
    PlotSceneData,
    normalize_plot_scene_data,
)
from dronalize.plot.theme import EdgeStyle, PlotTheme, ThemeName, get_plot_theme

try:
    import altair as alt
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="Scene plotting", extra="plot")

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.core.categories import AgentCategory, EdgeType
    from dronalize.core.scene import Scene
    from dronalize.io import SceneRecord

AspectMode = Literal["auto", "equal"]
"""Aspect-ratio modes supported by [`plot_scene`][dronalize.plot.plot_scene].

`"auto"` uses the default plotting dimensions unless explicit dimensions are
supplied, while `"equal"` computes plot bounds so one data unit in x matches
one data unit in y.
"""

ChartT: TypeAlias = alt.Chart | alt.LayerChart
FilterExpression: TypeAlias = str

_DEFAULT_WIDTH: Final[int] = 800
_DEFAULT_HEIGHT: Final[int] = 450
_MIN_DIMENSION: Final[int] = 240
_MAX_DIMENSION: Final[int] = 1200
_MAX_EQUAL_RATIO: Final[float] = 6.0
_MAX_TOOLTIP_AGENTS: Final[int] = 150
_FUTURE_PHASE_DASH: Final[tuple[int, int]] = (6, 12)

_FILTER_ALL: Final[str] = "All"

_SELECTED_FRAME_PARAM: Final[str] = "selected_frame"
_AGENT_TYPE_FILTER_PARAM: Final[str] = "agent_type_filter"
_SCREENING_FILTER_PARAM: Final[str] = "screening_filter"


@dataclass(slots=True, frozen=True)
class FrameControl:
    """Altair frame slider parameter and its derived filter expressions."""

    parameter: alt.Parameter
    visible_expr: FilterExpression
    current_expr: FilterExpression


@dataclass(slots=True, frozen=True)
class DropdownControl:
    """Altair dropdown parameter and its derived filter expression."""

    parameter: alt.Parameter
    expression: FilterExpression


@dataclass(slots=True, frozen=True)
class TrajectoryControls:
    """All interactive controls used by the trajectory layers."""

    selection: alt.Parameter
    focus: alt.Parameter
    frame: FrameControl
    category: DropdownControl
    screening: DropdownControl

    @property
    def visible_filters(self) -> tuple[FilterExpression, ...]:
        """Return filters shared by layers that respect the frame slider."""
        return (
            self.category.expression,
            self.screening.expression,
            self.frame.visible_expr,
        )

    @property
    def static_filters(self) -> tuple[FilterExpression, ...]:
        """Return filters shared by layers that ignore the frame slider."""
        return (
            self.category.expression,
            self.screening.expression,
        )

    @property
    def parameters(self) -> tuple[alt.Parameter, ...]:
        """Return Altair parameters that must be attached to the chart."""
        return (
            self.frame.parameter,
            self.category.parameter,
            self.screening.parameter,
        )


@dataclass(slots=True, frozen=True)
class MapStyleScale:
    """Resolved per-edge-type styling arrays used by Altair scales."""

    domain: list[str]
    colors: list[str]
    widths: list[float]
    dashes: list[tuple[int, ...]]


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
) -> ChartT:
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

    normalized = normalize_plot_scene_data(
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
) -> ChartT:
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
    params: list[alt.Parameter] = []

    if show_map and data.map_edges is not None and not data.map_edges.is_empty():
        layers.extend(
            _build_map_layers(
                data.map_edges,
                include_nodes=include_map_nodes,
                theme=theme,
            )
        )

    if not data.trajectories.is_empty():
        trajectory_layers, trajectory_params = _build_trajectory_layers(
            data.trajectories,
            theme=theme,
        )
        layers.extend(trajectory_layers)
        params.extend(trajectory_params)

    if not layers:
        return _empty_chart("Empty Scene Plot", theme=theme)

    chart: ChartT = alt.layer(*layers)
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
    trajectories: pl.DataFrame, *, theme: PlotTheme
) -> tuple[list[alt.Chart], tuple[alt.Parameter, ...]]:
    """Build Altair layers for scene trajectories."""
    unique_agents = trajectories["agent"].n_unique()
    controls = _trajectory_controls(trajectories)
    trajectory_paths = _trajectory_path_data(trajectories)
    agent_color = _agent_color_encoding(trajectories, theme=theme)
    line_tooltips = _trajectory_tooltips(include_point_details=unique_agents <= _MAX_TOOLTIP_AGENTS)
    marker_tooltips = _trajectory_tooltips(include_point_details=True)
    base = _trajectory_base_chart(
        trajectory_paths,
        controls=controls,
        agent_color=agent_color,
        tooltips=line_tooltips,
    )

    layers = [
        _trajectory_selection_targets(base, controls=controls),
        *_trajectory_line_layers(base, controls=controls),
        *_trajectory_marker_layers(
            trajectories, controls=controls, agent_color=agent_color, tooltips=marker_tooltips
        ),
    ]
    return layers, controls.parameters


def _trajectory_controls(trajectories: pl.DataFrame) -> TrajectoryControls:
    """Create the interactive controls shared by trajectory layers."""
    return TrajectoryControls(
        selection=alt.selection_point(fields=["agent"], on="click", clear="dblclick"),
        focus=alt.selection_point(fields=["agent"], on="click", clear="dblclick", empty=False),
        frame=_frame_control(trajectories),
        category=_dropdown_control(
            name=_AGENT_TYPE_FILTER_PARAM,
            label="Agent Type",
            field="agent_type",
            options=trajectories["agent_type"].unique().sort().to_list(),
        ),
        screening=_dropdown_control(
            name=_SCREENING_FILTER_PARAM,
            label="Screening",
            field="screening_status",
            options=_screening_statuses(trajectories),
        ),
    )


def _trajectory_base_chart(
    trajectory_paths: pl.DataFrame,
    *,
    controls: TrajectoryControls,
    agent_color: alt.Color,
    tooltips: list[alt.Tooltip],
) -> alt.Chart:
    """Create the shared base chart used by all trajectory layers."""
    return _filtered_chart(trajectory_paths, *controls.visible_filters).encode(
        x=alt.X("x:Q", title="X", scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title="Y", scale=alt.Scale(zero=False)),
        color=agent_color,
        order=alt.Order("frame:Q"),
        detail="phase_segment:N",
        tooltip=tooltips,
    )


def _trajectory_selection_targets(
    base: alt.Chart,
    *,
    controls: TrajectoryControls,
) -> alt.Chart:
    """Add an invisible interaction layer used for agent selection."""
    return base.mark_line(color="#000000", strokeWidth=14, opacity=0.001).add_params(
        controls.selection, controls.focus
    )


def _trajectory_line_layers(base: alt.Chart, *, controls: TrajectoryControls) -> list[alt.Chart]:
    """Create the trajectory line layers in back-to-front order."""
    trajectories = base.mark_line(strokeWidth=2.15).encode(
        strokeDash=_trajectory_phase_dash(show_legend=True),
        opacity=alt.when(controls.selection).then(alt.value(0.86)).otherwise(alt.value(0.06)),
    )

    focused_trajectory = (
        base
        .transform_filter(controls.focus)
        .mark_line(strokeWidth=2.9)
        .encode(
            strokeDash=_trajectory_phase_dash(show_legend=False),
            opacity=alt.value(1.0),
        )
    )

    return [trajectories, focused_trajectory]


def _trajectory_marker_layers(
    trajectories: pl.DataFrame,
    *,
    controls: TrajectoryControls,
    agent_color: alt.Color,
    tooltips: list[alt.Tooltip],
) -> list[alt.Chart]:
    """Create the marker layers that sit on top of the trajectory lines."""
    start_frames = trajectories.group_by("agent", maintain_order=True).head(1)

    start_markers = (
        _filtered_chart(start_frames, *controls.visible_filters)
        .mark_point(shape="circle", size=90, filled=True, fill="#2ecc71", fillOpacity=0.95)
        .encode(
            x="x:Q",
            y="y:Q",
            tooltip=tooltips,
            opacity=alt.when(controls.selection).then(alt.value(0.95)).otherwise(alt.value(0.15)),
        )
    )

    current_points = (
        _filtered_chart(trajectories, *controls.static_filters)
        .transform_filter(controls.frame.current_expr)
        .mark_point(shape="diamond", size=90, filled=True, fillOpacity=0.98)
        .encode(
            x="x:Q",
            y="y:Q",
            color=agent_color,
            tooltip=tooltips,
            opacity=alt.when(controls.selection).then(alt.value(0.95)).otherwise(alt.value(0.15)),
        )
    )

    return [start_markers, current_points]


def _trajectory_path_data(trajectories: pl.DataFrame) -> pl.DataFrame:
    """Prepare line-path rows so phase becomes a visible line style."""
    path_rows = trajectories.with_columns(pl.col("phase").alias("path_phase")).with_columns(
        _phase_segment_expr(phase_field="path_phase").alias("phase_segment")
    )
    if trajectories.is_empty():
        return path_rows

    transition_rows = (
        trajectories
        .sort(["agent", "segment", "frame"])
        .with_columns(
            pl.col("phase").shift(-1).over(["agent", "segment"]).alias("next_phase"),
            pl.col("frame").shift(-1).over(["agent", "segment"]).alias("next_frame"),
        )
        .filter(
            (pl.col("phase") == PHASE_HISTORY)
            & (pl.col("next_phase") == PHASE_FUTURE)
            & (pl.col("next_frame") == pl.col("frame") + 1)
        )
        .with_columns(pl.lit(PHASE_FUTURE).alias("path_phase"))
        .with_columns(_phase_segment_expr(phase_field="path_phase").alias("phase_segment"))
        .drop("next_phase", "next_frame")
    )

    return pl.concat([path_rows, transition_rows], how="vertical").sort([
        "agent",
        "segment",
        "frame",
        "path_phase",
    ])


def _phase_segment_expr(*, phase_field: str) -> pl.Expr:
    """Return a stable phase-aware segment identifier for Altair line grouping."""
    return pl.concat_str(
        pl.col("segment").cast(pl.String()),
        pl.lit("::"),
        pl.col(phase_field),
    )


def _trajectory_phase_dash(*, show_legend: bool) -> alt.StrokeDash:
    """Return the stroke-dash encoding used to distinguish history and future."""
    legend = alt.Legend(title="Trajectory Phase", orient="right") if show_legend else None
    return alt.StrokeDash(
        "path_phase:N",
        sort=[PHASE_HISTORY, PHASE_FUTURE],
        scale=alt.Scale(
            domain=[PHASE_HISTORY, PHASE_FUTURE],
            range=[(1, 0), _FUTURE_PHASE_DASH],
        ),
        legend=legend,
    )


def _trajectory_tooltips(
    *,
    include_point_details: bool,
) -> list[alt.Tooltip]:
    """Build a consistent tooltip payload for trajectory layers."""
    tooltips: list[alt.Tooltip] = [
        alt.Tooltip("agent:Q", title="Agent"),
        alt.Tooltip("agent_type:N", title="Type"),
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
    map_edges: pl.DataFrame, *, include_nodes: bool, theme: PlotTheme
) -> list[alt.Chart]:
    """Build Altair layers for a normalized edge dataframe."""
    style_scale = _map_style_scale(map_edges, theme=theme)

    lines = (
        alt
        .Chart(map_edges)
        .mark_rule(strokeCap="round")
        .encode(
            x=alt.X("x1:Q", scale=alt.Scale(zero=False), title="X", axis=alt.Axis(grid=False)),
            y=alt.Y("y1:Q", scale=alt.Scale(zero=False), title="Y", axis=alt.Axis(grid=False)),
            x2="x2:Q",
            y2="y2:Q",
            color=alt.Color(
                "edge_type:N",
                scale=alt.Scale(domain=style_scale.domain, range=style_scale.colors),
                legend=alt.Legend(title="Edge Type", symbolType="stroke", orient="right"),
            ),
            size=alt.Size(
                "edge_type:N",
                scale=alt.Scale(domain=style_scale.domain, range=style_scale.widths),
                legend=None,
            ),
            strokeDash=alt.StrokeDash(
                "edge_type:N",
                scale=alt.Scale(domain=style_scale.domain, range=style_scale.dashes),
                legend=None,
            ),
            opacity=alt.value(0.45),
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
        layers.extend([
            _map_node_layer(map_edges, x_field="x1", y_field="y1", theme=theme),
            _map_node_layer(map_edges, x_field="x2", y_field="y2", theme=theme),
        ])

    return layers


def _map_style_scale(map_edges: pl.DataFrame, *, theme: PlotTheme) -> MapStyleScale:
    """Resolve the map-edge styling arrays required by Altair scales."""
    domain = map_edges["edge_type"].unique().to_list()
    domain.sort()

    edge_styles = _edge_styles_by_name(theme.name)
    return MapStyleScale(
        domain=domain,
        colors=[edge_styles.get(edge_type, _DEFAULT_EDGE_STYLE).color for edge_type in domain],
        widths=[edge_styles.get(edge_type, _DEFAULT_EDGE_STYLE).width for edge_type in domain],
        dashes=[edge_styles.get(edge_type, _DEFAULT_EDGE_STYLE).dash for edge_type in domain],
    )


def _map_node_layer(
    map_edges: pl.DataFrame,
    *,
    x_field: str,
    y_field: str,
    theme: PlotTheme,
) -> alt.Chart:
    """Return one semi-transparent map-node layer."""
    return (
        alt
        .Chart(map_edges)
        .mark_circle(color=theme.map_node_color, size=18, opacity=0.35)
        .encode(
            x=alt.X(f"{x_field}:Q", scale=alt.Scale(zero=False)),
            y=alt.Y(f"{y_field}:Q", scale=alt.Scale(zero=False)),
            opacity=alt.value(0.28),
        )
    )


def _empty_chart(message: str, *, theme: PlotTheme) -> alt.Chart:
    """Create a simple text chart used for empty plotting inputs."""
    return (
        alt
        .Chart(pl.DataFrame())
        .mark_text(size=16, color=theme.title_color, fontWeight="bold")
        .encode(text=alt.value(message))
    )


def _filtered_chart(data: pl.DataFrame, *filters: FilterExpression) -> alt.Chart:
    """Return an Altair chart with all provided filters applied in order."""
    chart = alt.Chart(data)
    for filter_expr in filters:
        chart = chart.transform_filter(filter_expr)
    return chart


_DEFAULT_EDGE_STYLE: Final[EdgeStyle] = EdgeStyle(color="gray", width=1.0, dash=(1, 0))


@lru_cache(maxsize=2)
def _edge_styles_by_name(theme_name: ThemeName) -> dict[str, EdgeStyle]:
    """Return one theme's edge styles keyed by display name."""
    theme = get_plot_theme(theme_name)
    return {EDGE_TYPE_LABELS[value]: style for value, style in theme.edge_styles.items()}


def _agent_color_encoding(
    trajectories: pl.DataFrame,
    *,
    theme: PlotTheme,
) -> alt.Color:
    """Return a stable per-agent color encoding that does not change under filtering."""
    agent_ids = trajectories["agent"].unique().sort().to_list()
    scheme = theme.agent_color_scheme

    if isinstance(scheme, str):
        scale = alt.Scale(domain=agent_ids, scheme=scheme)  # pyright: ignore[reportArgumentType]
    else:
        palette = list(scheme)
        if not palette:
            msg = "agent_color_scheme cannot be empty."
            raise ValueError(msg)
        scale = alt.Scale(domain=agent_ids, range=palette)

    return alt.Color(
        "agent:N",
        legend=None,
        scale=scale,
    )


def _style_scene_chart(
    chart: ChartT, *, data: PlotSceneData, plot_width: int, plot_height: int, theme: PlotTheme
) -> ChartT:
    """Apply the shared theme used by scene plots."""
    chart = chart.properties(
        width=plot_width,
        height=plot_height,
        title=_scene_title(data),
    )
    return _apply_scene_theme(chart, theme=theme)


def _apply_scene_theme(chart: ChartT, *, theme: PlotTheme) -> ChartT:
    """Apply the shared chart-level Altair theme configuration."""
    return (
        chart
        .configure(background=theme.chart_background)
        .configure_view(
            fill=theme.plot_background,
            stroke=theme.grid_color,
            strokeOpacity=0.65,
        )
        .configure_title(
            color=theme.title_color,
            fontSize=18,
            subtitleFontSize=13,
            fontWeight="bold",
            offset=12,
            subtitleColor=theme.title_color,
        )
        .configure_axis(
            grid=False,
            domainColor=theme.grid_color,
            tickColor=theme.grid_color,
            labelColor=theme.axis_color,
            titleColor=theme.title_color,
            labelFontSize=12,
            titleFontSize=13,
        )
        .configure_legend(
            titleColor=theme.title_color,
            labelColor=theme.axis_color,
            fillColor=theme.plot_background,
            strokeColor=theme.grid_color,
            padding=10,
            orient="bottom",
            titleFontSize=14,
            labelFontSize=14,
            symbolSize=500,
            symbolStrokeWidth=5,
            symbolOpacity=1.0,
        )
    )


def _screening_statuses(trajectories: pl.DataFrame) -> list[str]:
    """Return screening statuses present in plotting order."""
    present = set(trajectories["screening_status"].unique().to_list())
    return [status for status in (SCREENING_PASSED, SCREENING_FAILED) if status in present]


def _frame_control(trajectories: pl.DataFrame) -> FrameControl:
    """Return the current-frame slider control."""
    frame_min, frame_max = trajectories["frame"].min(), trajectories["frame"].max()
    frame_min = int(frame_min) if isinstance(frame_min, (int, float)) else 0
    frame_max = int(frame_max) if isinstance(frame_max, (int, float)) else 0
    parameter = alt.param(
        name=_SELECTED_FRAME_PARAM,
        value=frame_max,
        bind=alt.binding_range(
            min=frame_min,
            max=frame_max,
            step=1,
            name="Frame ",
        ),
    )
    return FrameControl(
        parameter=parameter,
        visible_expr=f"datum.frame <= {_SELECTED_FRAME_PARAM}",
        current_expr=f"datum.frame == {_SELECTED_FRAME_PARAM}",
    )


def _dropdown_control(
    *,
    name: str,
    label: str,
    field: str,
    options: list[str],
) -> DropdownControl:
    """Create a dropdown filter control and its Vega expression."""
    parameter = alt.param(
        name=name,
        value=_FILTER_ALL,
        bind=alt.binding_select(
            options=[_FILTER_ALL, *options],
            name=f"{label} ",
        ),
    )
    return DropdownControl(
        parameter=parameter,
        expression=f"{name} == '{_FILTER_ALL}' || datum.{field} == {name}",
    )


def _scene_title(data: PlotSceneData) -> alt.TitleParams:
    """Return the chart title and summary subtitle for one plotted scene."""
    agent_count = 0
    category_count = 0
    passed_count = 0
    failed_count = 0
    frame_range = "frames: n/a"

    if not data.trajectories.is_empty():
        agent_summary = data.trajectories.select(
            "agent",
            "agent_type",
            "screening_status",
        ).unique(subset="agent")
        agent_count = agent_summary.height
        category_count = agent_summary["agent_type"].n_unique()
        passed_count = int((agent_summary["screening_status"] == SCREENING_PASSED).sum())
        failed_count = int((agent_summary["screening_status"] == SCREENING_FAILED).sum())
        frame_min = data.trajectories["frame"].min()
        frame_max = data.trajectories["frame"].max()
        frame_range = f"frames: {frame_min}-{frame_max}"

    edge_count = 0 if data.map_edges is None else data.map_edges.height
    subtitle = (
        f"{agent_count} agents | {category_count} categories | "
        f"{passed_count} passed / {failed_count} failed | "
        f"{edge_count} map edges | {frame_range}"
    )
    return alt.TitleParams(
        text=f"Scene {data.scene_number}",
        subtitle=[subtitle],
        anchor="start",
    )
