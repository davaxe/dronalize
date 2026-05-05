"""Options for compiling semantic geometry into a graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dronalize.core.categories import EdgeType


@dataclass(frozen=True, slots=True)
class MapBuildOptions:
    """Global sampling and edge-remapping options for map compilation."""

    min_distance: float
    interp_distance: float
    edge_remap: dict[EdgeType, EdgeType] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate distance parameters after initialization."""
        if self.interp_distance <= 0.0:
            msg = "interp_distance must be greater than 0."
            raise ValueError(msg)
        if not (0.0 <= self.min_distance <= self.interp_distance):
            msg = (
                "min_distance must be in the range "
                f"[0, interp_distance] ([0, {self.interp_distance}])."
            )
            raise ValueError(msg)

    @classmethod
    def from_distances(
        cls,
        min_distance: float | None,
        interp_distance: float | None,
        *,
        edge_remap: Mapping[EdgeType, EdgeType] | None = None,
    ) -> MapBuildOptions:
        """Create validated options from nullable distance inputs."""
        resolved_min_distance = 0.0 if min_distance is None else min_distance
        resolved_interp_distance = np.inf if interp_distance is None else interp_distance
        return cls(
            min_distance=resolved_min_distance,
            interp_distance=resolved_interp_distance,
            edge_remap={} if edge_remap is None else dict(edge_remap),
        )
