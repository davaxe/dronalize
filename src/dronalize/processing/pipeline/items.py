"""Typed payloads passed between runtime trajectory processing stages."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from dronalize.processing.loading.models import MapReference

if TYPE_CHECKING:
    import polars as pl

    from dronalize.core.categories import DatasetSplit
    from dronalize.processing.loading.models import DatasetSource
    from dronalize.processing.screening.screen import ScreeningResult


@dataclass(frozen=True, slots=True)
class TrajectoryPipelineItem:
    """One candidate scene plus structured metadata emitted by processing stages."""

    source: DatasetSource[Any]
    stable_identifier: object
    frame: pl.DataFrame
    map_reference: MapReference = field(default_factory=MapReference)
    screening: ScreeningResult | None = None
    split_assignment: DatasetSplit | None = None

    def with_frame(self, frame: pl.DataFrame) -> TrajectoryPipelineItem:
        """Return this item with a replaced frame."""
        return replace(self, frame=frame)

    def with_screening(self, screening: ScreeningResult) -> TrajectoryPipelineItem:
        """Return this item with structured screening output attached."""
        return replace(self, frame=screening.frame, screening=screening)

    def with_split_assignment(
        self, split_assignment: DatasetSplit | None
    ) -> TrajectoryPipelineItem:
        """Return this item with an output split assignment attached."""
        return replace(self, split_assignment=split_assignment)
