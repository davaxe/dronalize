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

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING

from dronalize.core.categories import EdgeType

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dronalize.core.graph_builder import Point


class Argoverse1Map:
    """Represents the Argoverse map, containing lane segments and nodes."""

    def __init__(self, path: Path) -> None:
        """Initialize the ArgoverseMap with the path to the XML file."""
        self.path: Path = path
        self.lane_segments: dict[int, LaneSegment] = {}
        self.nodes: dict[int, Point] = {}
        self._parsed: bool = False

    @classmethod
    def from_xml_file(cls, path: Path) -> Argoverse1Map:
        """Create an `ArgoverseMap` instance from an XML file."""
        return cls(path)

    def is_parsed(self) -> bool:
        """Check if the map has been parsed."""
        return self._parsed

    def parse(self) -> None:
        """Parse the XML file and extract nodes and lane segments."""
        if self._parsed:
            return

        tree = ET.parse(self.path)
        root = tree.getroot()

        all_graph_nodes: dict[int, Point] = {}
        # Mapping from original ID in the XML to new sequential ID.
        id_map: dict[int, int] = {}
        next_id = 0
        for child in root:
            if child.tag == "node":
                base_node = Node.from_xml_element(child)
                point: Point = (base_node.x, base_node.y)
                id_map[base_node.id] = next_id
                all_graph_nodes[next_id] = point
                next_id += 1
            elif child.tag == "way":
                lane_segment = LaneSegment.from_xml_element(child)
                self.lane_segments[lane_segment.id] = lane_segment
            else:
                msg = f"Unknown XML item: {child.tag} with attributes {child.attrib}"
                raise ValueError(msg)

        for lane_segment in self.lane_segments.values():
            # Map the lane segment's node IDs to the new node IDs
            lane_segment.map_node_ids(lambda node_id: id_map[node_id])

        # Dictionary respect ordering, which means that the order of nodes is
        # preserved
        self.nodes = all_graph_nodes
        self._parsed = True


class TurnType(IntEnum):
    """Represents the turn direction of a lane segment (way)."""

    NONE = auto()
    LEFT = auto()
    RIGHT = auto()


@dataclass
class Node:
    """Represents a node in the Argoverse map."""

    id: int
    x: float
    y: float

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Node:
        """Create a `Node` instance from a dictionary."""
        return cls(
            id=int(data["id"]),
            x=float(data["x"]),
            y=float(data["y"]),
        )

    @classmethod
    def from_xml_element(cls, element: ET.Element) -> Node:
        """Create a `Node` instance from an XML element."""
        node_fields = element.attrib
        return cls.from_dict(node_fields)


@dataclass
class LaneSegment:
    """Represents a lane segment (way) in the Argoverse map."""

    id: int
    node_ids: list[int]
    l_neighbor_id: int | None = None
    r_neighbor_id: int | None = None
    turn_direction: TurnType = TurnType.NONE
    is_intersection: bool = False
    has_traffic_control: bool = False
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)

    @classmethod
    def from_xml_element(cls, element: ET.Element) -> LaneSegment:
        """Create a `LaneSegment` instance from an XML element."""
        lane_id = int(element.attrib["lane_id"])
        segment = cls._default()
        segment.id = lane_id
        for sub_element in element:
            # Cast inspired from argoverse1 official code
            way_field: list[tuple[str, str]] = list(sub_element.items())
            field_name = way_field[0][0]
            if field_name == "ref":
                node_id = int(way_field[0][1])
                segment.node_ids.append(node_id)
                continue

            match way_field[0], way_field[1]:
                case ("k", "is_intersection"), ("v", v):
                    segment.is_intersection = v == "True"
                case ("k", "has_traffic_control"), ("v", v):
                    segment.has_traffic_control = v == "True"
                case ("k", "turn_direction"), ("v", v):
                    segment.turn_direction = TurnType[v]
                case ("k", "l_neighbor_id"), ("v", v):
                    segment.l_neighbor_id = None if v == "None" else int(v)
                case ("k", "r_neighbor_id"), ("v", v):
                    segment.r_neighbor_id = None if v == "None" else int(v)
                case ("k", "predecessor"), ("v", v):
                    segment.predecessors.append(int(v))
                case ("k", "successor"), ("v", v):
                    segment.successors.append(int(v))
                case _:
                    ...

        return segment

    def map_node_ids(self, map_fn: Callable[[int], int]) -> None:
        """Map the node IDs using a provided function."""
        self.node_ids = [map_fn(node_id) for node_id in self.node_ids]

    def get_edge_type(self) -> EdgeType:
        """Determine the edge type based on the lane segment's properties."""
        ln, rn, inter = (
            self.l_neighbor_id is not None,
            self.r_neighbor_id is not None,
            self.is_intersection,
        )
        if ln and rn:
            return EdgeType.LINE_THIN
        if not ln and not rn and not inter:
            return EdgeType.LINE_THIN
        if not inter:
            return EdgeType.CURB
        return EdgeType.NONE

    def get_border_edge_types(self) -> tuple[EdgeType, EdgeType]:
        """Determine the border edge types based on the lane segment's neighbors.

        Returns
        -------
        left_edge_type : EdgeType
            Edge type for the left border.
        right_edge_type : EdgeType
            Edge type for the right border.

        """
        ln, rn = (
            self.l_neighbor_id is not None,
            self.r_neighbor_id is not None,
        )
        left_edge_type = EdgeType.VIRTUAL if ln else EdgeType.CURB
        right_edge_type = EdgeType.VIRTUAL if rn else EdgeType.CURB

        if self.is_intersection:
            left_edge_type, right_edge_type = EdgeType.VIRTUAL, EdgeType.VIRTUAL

        return left_edge_type, right_edge_type

    @classmethod
    def _default(cls) -> LaneSegment:
        """Create a default `LaneSegment` instance."""
        return cls(
            id=0,
            node_ids=[],
            l_neighbor_id=None,
            r_neighbor_id=None,
            turn_direction=TurnType.NONE,
            is_intersection=False,
            has_traffic_control=False,
            successors=[],
            predecessors=[],
        )
