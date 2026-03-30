from __future__ import annotations

from enum import IntEnum
from typing import Final


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
    def from_str(cls, type_str: str | None, subtype: str | None = None) -> EdgeType:
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
