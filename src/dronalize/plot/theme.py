"""Theme definitions for scene plotting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from dronalize.core.categories import EdgeType

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
    trajectory_halo_color: str
    map_node_color: str
    agent_color_scheme: str
    edge_styles: dict[int, EdgeStyle]


_BASE_EDGE_STYLES: dict[int, EdgeStyle] = {
    EdgeType.NONE.value: EdgeStyle("#aea89c", 0.55, (1, 0)),
    EdgeType.ROAD_BORDER.value: EdgeStyle("#52616d", 1.9, (1, 0)),
    EdgeType.CURB.value: EdgeStyle("#455765", 2.1, (1, 0)),
    EdgeType.REGULATORY.value: EdgeStyle("#bc6258", 1.1, (1, 0)),
    EdgeType.VIRTUAL.value: EdgeStyle("#7fa2b8", 0.9, (3, 3)),
    EdgeType.LINE_THIN.value: EdgeStyle("#919da7", 1.0, (1, 0)),
    EdgeType.LINE_THIN_DASHED.value: EdgeStyle("#919da7", 1.0, (5, 5)),
    EdgeType.LINE_THICK.value: EdgeStyle("#72808a", 1.7, (1, 0)),
    EdgeType.LINE_THICK_DASHED.value: EdgeStyle("#72808a", 1.7, (5, 5)),
    EdgeType.PEDESTRIAN_MARKING.value: EdgeStyle("#d29f5d", 1.3, (5, 5)),
    EdgeType.BIKE_MARKING.value: EdgeStyle("#5f9170", 1.3, (5, 5)),
    EdgeType.GUARD_RAIL.value: EdgeStyle("#8a76a1", 1.7, (1, 0)),
    EdgeType.STOP.value: EdgeStyle("#c05b4d", 2.3, (1, 0)),
    EdgeType.LINE_THIN_DOUBLE.value: EdgeStyle("#c0a45b", 1.7, (1, 0)),
    EdgeType.LINE_THIN_DOUBLE_DASHED.value: EdgeStyle("#c0a45b", 1.7, (5, 5)),
}


def _edge_styles(overrides: dict[int, str] | None = None) -> dict[int, EdgeStyle]:
    """Return themed edge styles with optional color overrides."""
    if overrides is None:
        return dict(_BASE_EDGE_STYLES)
    return {
        value: replace(style, color=overrides.get(value, style.color))
        for value, style in _BASE_EDGE_STYLES.items()
    }


LIGHT_THEME = PlotTheme(
    name="light",
    chart_background="#f5f1e8",
    plot_background="#fcfaf5",
    grid_color="#d7d0c4",
    axis_color="#6b7280",
    title_color="#27313a",
    trajectory_halo_color="#fffdf8",
    map_node_color="#5d6b75",
    agent_color_scheme="tableau20",
    edge_styles=_edge_styles(),
)


DARK_THEME = PlotTheme(
    name="dark",
    chart_background="#0f172a",
    plot_background="#111827",
    grid_color="#334155",
    axis_color="#cbd5e1",
    title_color="#f8fafc",
    trajectory_halo_color="#e2e8f0",
    map_node_color="#94a3b8",
    agent_color_scheme="tableau20",
    edge_styles=_edge_styles({
        EdgeType.NONE.value: "#596273",
        EdgeType.ROAD_BORDER.value: "#90a4b8",
        EdgeType.CURB.value: "#a3b5c7",
        EdgeType.REGULATORY.value: "#f28b82",
        EdgeType.VIRTUAL.value: "#78a9ff",
        EdgeType.LINE_THIN.value: "#8b98a6",
        EdgeType.LINE_THIN_DASHED.value: "#8b98a6",
        EdgeType.LINE_THICK.value: "#b0becb",
        EdgeType.LINE_THICK_DASHED.value: "#b0becb",
        EdgeType.PEDESTRIAN_MARKING.value: "#f6bd60",
        EdgeType.BIKE_MARKING.value: "#7bd389",
        EdgeType.GUARD_RAIL.value: "#c4b5fd",
        EdgeType.STOP.value: "#f87171",
        EdgeType.LINE_THIN_DOUBLE.value: "#f4d35e",
        EdgeType.LINE_THIN_DOUBLE_DASHED.value: "#f4d35e",
    }),
)


_THEMES: dict[ThemeName, PlotTheme] = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
}


def get_plot_theme(theme: ThemeName | PlotTheme) -> PlotTheme:
    """Resolve one built-in or custom plot theme."""
    if isinstance(theme, str):
        return _THEMES[theme]
    return theme
