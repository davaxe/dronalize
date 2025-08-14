# Copyright 2024-2025, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from enum import IntEnum
from typing import Final, Literal, TypedDict


class EdgeType(IntEnum):
    """Enumeration of edge types in the lane graph."""

    NONE = 1
    ROAD_BORDER = 2
    CURB = 3
    REGULATORY = 4
    VIRTUAL = 5
    LINE_THIN = 6
    LINE_THIN_DASHED = 7
    LINE_THICK = 8
    LINE_THICK_DASHED = 9
    PEDESTRIAN_MARKING = 10
    BIKE_MARKING = 11
    GUARD_RAIL = 12
    STOP = 13
    LINE_THIN_DOUBLE = 14
    LINE_THIN_DOUBLE_DASHED = 15

    @classmethod
    def from_str(
        cls,
        type_str: str | None,
        subtype: str | None = None,
    ) -> EdgeType:
        """Convert string representation to EdgeType."""
        if type_str is None:
            return cls.NONE

        # First check high-priority mappings
        type_map: dict[str, EdgeType] = _STR_TO_EDGE

        # Check if it's a high-priority type first
        edge_type = type_map.get(type_str)
        if edge_type is not None:
            return edge_type

        # Only then check for line types
        if type_str == "line_thin":
            return cls.LINE_THIN_DASHED if subtype == "dashed" else cls.LINE_THIN
        if type_str == "line_thick":
            return cls.LINE_THICK_DASHED if subtype == "dashed" else cls.LINE_THICK

        return cls.NONE

    def node_style(self) -> NodeStyle:
        """Get the node style associated with this edge type."""
        return NODE_STYLE_MAPPING.get(self, NODE_STYLE_MAPPING[EdgeType.NONE])

    def edge_style(self) -> EdgeStyle:
        """Get the edge style associated with this edge type."""
        return EDGE_STYLE_MAPPING.get(self, EDGE_STYLE_MAPPING[EdgeType.NONE])


# Constructed once and used as a constant mapping
_STR_TO_EDGE: Final[dict[str, EdgeType]] = {
    "road_border": EdgeType.ROAD_BORDER,
    "curbstone": EdgeType.CURB,
    "stop_line": EdgeType.STOP,
    "regulatory_element": EdgeType.REGULATORY,
    "virtual": EdgeType.VIRTUAL,
    "pedestrian_marking": EdgeType.PEDESTRIAN_MARKING,
    "bike_marking": EdgeType.BIKE_MARKING,
    "guard_rail": EdgeType.GUARD_RAIL,
    "fence": EdgeType.ROAD_BORDER,
    "wall": EdgeType.ROAD_BORDER,
}


# TypedDict used for explicit typing of edge and node styles
class EdgeStyle(TypedDict):
    """Defines the plotting style for edges and nodes in the map graph.

    For convenience, the key names match the matplotlib parameters. This allows
    using **EdgeStyle as keyword arguments in matplotlib functions.

    Note: The `dashed` key will have priority over `linestyle` if both are
    provided in a in matplotlib function call.
    """

    color: str
    linewidth: float
    linestyle: LineStyles
    dashes: list[int]


class NodeStyle(TypedDict):
    """Defines the plotting style for nodes in the map graph.

    For convenience, the key names match the matplotlib parameters.
    This allows using **NodeStyle as keyword arguments in matplotlib functions.
    """

    color: str
    s: int  # Size of the node marker
    marker: MarkerStyles


EDGE_STYLE_MAPPING: dict[EdgeType, EdgeStyle] = {
    EdgeType.NONE: {
        "color": "purple",
        "linewidth": 0.5,
        "linestyle": "solid",
        "dashes": [],
    },
    EdgeType.ROAD_BORDER: {
        "color": "black",
        "linewidth": 1.0,
        "linestyle": "solid",
        "dashes": [],
    },
    EdgeType.CURB: {
        "color": "black",
        "linewidth": 1.0,
        "linestyle": "solid",
        "dashes": [],
    },
    EdgeType.REGULATORY: {
        "color": "tab:orange",
        "linewidth": 1.0,
        "linestyle": "solid",
        "dashes": [],
    },
    EdgeType.VIRTUAL: {
        "color": "tab:blue",
        "linewidth": 1.0,
        "linestyle": "dotted",
        "dashes": [2, 5],
    },
    EdgeType.LINE_THIN: {
        "color": "grey",
        "linewidth": 1.0,
        "linestyle": "solid",
        "dashes": [],
    },
    EdgeType.LINE_THIN_DASHED: {
        "color": "grey",
        "linewidth": 1.0,
        "linestyle": "dashed",
        "dashes": [10, 10],
    },
    EdgeType.LINE_THICK: {
        "color": "grey",
        "linewidth": 2.0,
        "linestyle": "solid",
        "dashes": [],
    },
    EdgeType.LINE_THICK_DASHED: {
        "color": "white",
        "linewidth": 2.0,
        "linestyle": "dashed",
        "dashes": [10, 10],
    },
    EdgeType.PEDESTRIAN_MARKING: {
        "color": "purple",
        "linewidth": 1.0,
        "linestyle": "dashed",
        "dashes": [5, 10],
    },
    EdgeType.BIKE_MARKING: {
        "color": "white",
        "linewidth": 1.0,
        "linestyle": "dashed",
        "dashes": [5, 10],
    },
    EdgeType.GUARD_RAIL: {
        "color": "black",
        "linewidth": 1.0,
        "linestyle": "solid",
        "dashes": [],
    },
    EdgeType.STOP: {
        "color": "red",
        "linewidth": 1.0,
        "linestyle": "solid",
        "dashes": [],
    },
}

NODE_STYLE_MAPPING: dict[EdgeType, NodeStyle] = {
    EdgeType.NONE: {"color": "grey", "s": 20, "marker": "o"},
    EdgeType.ROAD_BORDER: {"color": "tab:red", "s": 20, "marker": "o"},
    EdgeType.CURB: {"color": "tab:green", "s": 20, "marker": "H"},
    EdgeType.REGULATORY: {"color": "tab:orange", "s": 30, "marker": "o"},
    EdgeType.VIRTUAL: {"color": "tab:blue", "s": 20, "marker": "o"},
    EdgeType.LINE_THIN: {"color": "grey", "s": 20, "marker": "o"},
    EdgeType.LINE_THIN_DASHED: {"color": "grey", "s": 20, "marker": "o"},
    EdgeType.LINE_THICK: {"color": "grey", "s": 25, "marker": "o"},
    EdgeType.LINE_THICK_DASHED: {"color": "grey", "s": 25, "marker": "o"},
    EdgeType.PEDESTRIAN_MARKING: {"color": "yellow", "s": 20, "marker": "o"},
    EdgeType.BIKE_MARKING: {"color": "cyan", "s": 25, "marker": "o"},
    EdgeType.GUARD_RAIL: {"color": "black", "s": 20, "marker": "o"},
    EdgeType.STOP: {"color": "red", "s": 25, "marker": "o"},
}

LineStyles = Literal[
    "-",
    "--",
    "-.",
    ":",
    "None",
    " ",
    "",
    "solid",
    "dashed",
    "dashdot",
    "dotted",
]

MarkerStyles = Literal[
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    # There are more available: https://matplotlib.org/stable/api/markers_api.html
]
