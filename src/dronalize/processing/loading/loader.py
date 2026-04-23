"""Loader-side data structures for source-to-scene processing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic

from dronalize.core.typing import SourceId, SourceT

if TYPE_CHECKING:
    import polars as pl

    from dronalize.core.categories import DatasetSplit
    from dronalize.processing.maps.resolver import MapKey


@dataclass(slots=True, frozen=True)
class MapBinding:
    """Loader-side map attachment carried alongside ingested or processed data.

    The binding intentionally stays lightweight. Datasets can attach a stable
    map key and, when map data is already read together with trajectories, a
    serialized map payload for their loader-specific `resolve_map()`
    implementation.
    """

    map_key: MapKey = None
    """Stable map identifier for the scene, if one is known at ingest time."""
    map_payload: bytes | None = None
    """Serialized map payload already available from trajectory ingestion."""


@dataclass(slots=True, frozen=True)
class LoadedSourceData:
    """One source-derived lazy frame plus any scene-level map binding."""

    frame: pl.LazyFrame
    map_binding: MapBinding = field(default_factory=MapBinding)
    predefined_split: DatasetSplit | None = None


@dataclass(slots=True, frozen=True)
class Source(Generic[SourceT]):
    """Lightweight unit of raw input that yields one or more scenes."""

    identifier: SourceId
    """Stable identifier for the source, e.g., file name, URL, database key."""
    data: SourceT
    """Lightweight source payload, usually a path or small tuple of lookup values."""
    predefined_split: DatasetSplit | None = None
    """Predefined split, if any."""
    map_key: MapKey = None
    """Optional map key associated with this source."""

    def with_predefined_split(self, split_assignment: DatasetSplit | None) -> Source[SourceT]:
        """Return a copy with a concrete split assignment."""
        return replace(self, predefined_split=split_assignment)
