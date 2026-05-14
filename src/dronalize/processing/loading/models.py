"""Loader-side data structures for DatasetSource-to-scene processing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Generic

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from dronalize.core.typing import SourceId, SourceT

if TYPE_CHECKING:
    import polars as pl

    from dronalize.core.categories import DatasetSplit
    from dronalize.processing.maps import MapKey


@dataclass(slots=True, frozen=True)
class MapReference:
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
class LoadedSourceFrame:
    """One DatasetSource-derived lazy frame plus any scene-level map binding."""

    frame: pl.LazyFrame
    map_binding: MapReference = field(default_factory=MapReference)
    predefined_split: DatasetSplit | None = None


@dataclass(slots=True, frozen=True)
class DatasetRunResources:
    """Shared resources prepared once for a processing run.

    The main current use case is shared-memory map lookup tables, but the
    container stays intentionally generic so datasets can grow into additional
    run-scoped resources without reintroducing loader-global state.
    """

    shared_maps: dict[MapKey, str] | str | None = None

    @classmethod
    def empty(cls) -> DatasetRunResources:
        """Create an empty DatasetRunResources instance."""
        return cls()


@dataclass(slots=True, frozen=True)
class DatasetSource(Generic[SourceT]):
    """Lightweight unit of raw input that yields one or more scenes."""

    identifier: SourceId
    """Stable identifier for the source, e.g., file name, URL, database key."""
    payload: SourceT
    """Lightweight DatasetSource payload, usually a path or small tuple of lookup values."""
    predefined_split: DatasetSplit | None = None
    """Predefined split, if any."""
    map_key: MapKey = None
    """Optional map key associated with this DatasetSource."""

    def with_predefined_split(
        self, split_assignment: DatasetSplit | None
    ) -> DatasetSource[SourceT]:
        """Return a copy with a concrete split assignment."""
        return replace(self, predefined_split=split_assignment)


class DatasetOptionsModel(BaseModel):
    """Base model for dataset-specific declarative config."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def parse(cls, payload: dict[str, object] | None = None) -> Self:
        """Validate one plain dataset-owned config mapping."""
        return cls(**(payload or {}))


class NoDatasetOptions(DatasetOptionsModel):
    """Empty dataset config for datasets without dataset-owned settings."""
