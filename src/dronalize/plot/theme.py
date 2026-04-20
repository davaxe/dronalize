"""Theme definitions for scene plotting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

from dronalize.core.categories import EdgeType

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ThemeName = Literal["light", "dark"]


@dataclass(slots=True, frozen=True)
class EdgeStyle:
    """Styling for one map-edge type."""

    color: str
    width: float
    dash: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class PlotTheme:
    """Visual theme used by scene plots."""

    name: str
    chart_background: str
    plot_background: str
    grid_color: str
    axis_color: str
    title_color: str
    subtitle_color: str
    map_node_color: str
    agent_color_scheme: str | Sequence[str]
    edge_styles: Mapping[int, EdgeStyle]


_BASE_EDGE_STYLES: Final[dict[int, EdgeStyle]] = {
    EdgeType.NONE.value: EdgeStyle("#c7ced6", 0.55, (1, 0)),
    EdgeType.ROAD_BORDER.value: EdgeStyle("#607c8a", 1.9, (1, 0)),
    EdgeType.CURB.value: EdgeStyle("#36464e", 2.1, (1, 0)),
    EdgeType.REGULATORY.value: EdgeStyle("#ff6e42", 1.1, (1, 0)),
    EdgeType.VIRTUAL.value: EdgeStyle("#4287ff", 0.8, (3, 3)),
    EdgeType.LINE_THIN.value: EdgeStyle("#a0a8b2", 1.0, (1, 0)),
    EdgeType.LINE_THIN_DASHED.value: EdgeStyle("#a0a8b2", 1.0, (5, 5)),
    EdgeType.LINE_THICK.value: EdgeStyle("#70808f", 1.7, (1, 0)),
    EdgeType.LINE_THICK_DASHED.value: EdgeStyle("#70808f", 1.7, (5, 5)),
    EdgeType.PEDESTRIAN_MARKING.value: EdgeStyle("#d19d00", 1.3, (5, 5)),
    EdgeType.BIKE_MARKING.value: EdgeStyle("#1c7d4d", 1.3, (5, 5)),
    EdgeType.GUARD_RAIL.value: EdgeStyle("#8a63d2", 1.7, (1, 0)),
    EdgeType.STOP.value: EdgeStyle("#d52a2a", 2.3, (1, 0)),
    EdgeType.LINE_THIN_DOUBLE.value: EdgeStyle("#b8a500", 1.7, (1, 0)),
    EdgeType.LINE_THIN_DOUBLE_DASHED.value: EdgeStyle("#b8a500", 1.7, (5, 5)),
}


def _edge_styles(overrides: Mapping[int, str] | None = None) -> dict[int, EdgeStyle]:
    """Return themed edge styles with optional color overrides."""
    if overrides is None:
        return dict(_BASE_EDGE_STYLES)

    return {
        value: replace(style, color=overrides.get(value, style.color))
        for value, style in _BASE_EDGE_STYLES.items()
    }


LIGHT_AGENT_PALETTE = [
    "#356EAD",
    "#D97A16",
    "#3D8B40",
    "#C74646",
    "#8A60B8",
    "#B99817",
    "#2F8F8F",
    "#CC5A8A",
    "#8B5E3C",
    "#6E757C",
    "#4F9DA6",
    "#A65F95",
]

DARK_AGENT_PALETTE = [
    "#6BA3D6",
    "#FFAE57",
    "#7BC96F",
    "#FF7F7F",
    "#C6A0F6",
    "#F2D46B",
    "#7FD1CC",
    "#FF9FC1",
    "#C49A7A",
    "#C0C6CC",
    "#8DD3E0",
    "#D98BC3",
]

LIGHT_THEME: Final[PlotTheme] = PlotTheme(
    name="light",
    chart_background="#ffffff",
    plot_background="#f5f5f5",
    grid_color="#d7dde5",
    axis_color="#36464e",
    title_color="#111827",
    subtitle_color="#475569",
    map_node_color="#607c8a",
    agent_color_scheme=LIGHT_AGENT_PALETTE,
    edge_styles=_edge_styles({
        EdgeType.NONE.value: "#d2d8df",
        EdgeType.ROAD_BORDER.value: "#546d78",
        EdgeType.CURB.value: "#36464e",
        EdgeType.REGULATORY.value: "#ff6e42",
        EdgeType.VIRTUAL.value: "#4287ff",
        EdgeType.LINE_THIN.value: "#a7b0ba",
        EdgeType.LINE_THIN_DASHED.value: "#a7b0ba",
        EdgeType.LINE_THICK.value: "#70808f",
        EdgeType.LINE_THICK_DASHED.value: "#70808f",
        EdgeType.PEDESTRIAN_MARKING.value: "#d19d00",
        EdgeType.BIKE_MARKING.value: "#1c7d4d",
        EdgeType.GUARD_RAIL.value: "#7c4dff",
        EdgeType.STOP.value: "#d52a2a",
        EdgeType.LINE_THIN_DOUBLE.value: "#b8a500",
        EdgeType.LINE_THIN_DOUBLE_DASHED.value: "#b8a500",
    }),
)


DARK_THEME: Final[PlotTheme] = PlotTheme(
    name="dark",
    chart_background="#0d1117",
    plot_background="#161b22",
    grid_color="#30363d",
    axis_color="#9da7b3",
    title_color="#f0f6fc",
    subtitle_color="#c9d1d9",
    map_node_color="#8b949e",
    agent_color_scheme=DARK_AGENT_PALETTE,
    edge_styles=_edge_styles({
        EdgeType.NONE.value: "#484f58",
        EdgeType.ROAD_BORDER.value: "#8b949e",
        EdgeType.CURB.value: "#c9d1d9",
        EdgeType.REGULATORY.value: "#ff7b72",
        EdgeType.VIRTUAL.value: "#79c0ff",
        EdgeType.LINE_THIN.value: "#6e7681",
        EdgeType.LINE_THIN_DASHED.value: "#6e7681",
        EdgeType.LINE_THICK.value: "#a5b4c3",
        EdgeType.LINE_THICK_DASHED.value: "#a5b4c3",
        EdgeType.PEDESTRIAN_MARKING.value: "#e3b341",
        EdgeType.BIKE_MARKING.value: "#7ee787",
        EdgeType.GUARD_RAIL.value: "#d2a8ff",
        EdgeType.STOP.value: "#f85149",
        EdgeType.LINE_THIN_DOUBLE.value: "#f2cc60",
        EdgeType.LINE_THIN_DOUBLE_DASHED.value: "#f2cc60",
    }),
)


_THEMES: Final[dict[ThemeName, PlotTheme]] = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
}


def get_plot_theme(theme: ThemeName | PlotTheme) -> PlotTheme:
    """Resolve one built-in or custom plot theme."""
    if isinstance(theme, str):
        return _THEMES[theme]
    return theme
