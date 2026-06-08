"""Loader-side data structures for DatasetSource-to-scene processing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self, override

from dronalize.core.maps import MapGraph
from dronalize.core.scene import Scene
from dronalize.core.typing import SourceId, SourceT

if TYPE_CHECKING:
    import polars as pl

    from dronalize.core.categories import DatasetSplit


@dataclass(slots=True, frozen=True)
class MapReference:
    """Loader-side map reference carried alongside ingested or processed data.

    The reference intentionally stays lightweight. Datasets can provide a stable
    map key and, when map data is already read together with trajectories, a
    serialized map payload for their loader-specific `resolve_map()`
    implementation.
    """

    map_key: str | None = None
    """Stable map identifier for the scene, if one is known at ingest time."""
    map_payload: bytes | None = None
    """Serialized map payload already available from trajectory ingestion."""


@dataclass(slots=True, frozen=True)
class LoadedSourceFrame:
    """One DatasetSource-derived lazy frame plus any scene-level map reference."""

    frame: pl.LazyFrame
    map_reference: MapReference = field(default_factory=MapReference)
    source_split: DatasetSplit | None = None


@dataclass(slots=True, frozen=True)
class DatasetRunResources:
    """Shared resources prepared once for a processing run."""

    map_provider: MapProvider | None = None


@dataclass(slots=True, frozen=True)
class DatasetSource(Generic[SourceT]):
    """Lightweight unit of raw input that yields one or more scenes."""

    identifier: SourceId
    """Stable identifier for the source, e.g., file name, URL, database key."""
    payload: SourceT
    """Lightweight DatasetSource payload, usually a path or small tuple of lookup values."""
    source_split: DatasetSplit | None = None
    """Native/source split carried before output split assignment, if any."""
    map_key: str | None = None
    """Optional map key associated with this DatasetSource."""

    def with_source_split(self, source_split: DatasetSplit | None) -> DatasetSource[SourceT]:
        """Return a copy with a source split."""
        return replace(self, source_split=source_split)


class LoaderOptionsModel(BaseModel):
    """Base model for dataset-specific loader options."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def parse(cls, payload: dict[str, object] | None = None) -> Self:
        """Validate one plain dataset-owned config mapping."""
        return cls(**(payload or {}))


class NoLoaderOptions(LoaderOptionsModel):
    """Empty loader-options model for datasets without dataset-owned settings."""


MapExtractor: TypeAlias = Callable[[Scene, MapGraph], MapGraph]


class MapProvider(Protocol):
    """Resolve map graphs for materialized scenes."""

    def resolve(self, scene: Scene, reference: MapReference) -> MapGraph | None:
        """Resolve the map for *scene* using loader-supplied map reference data."""
        ...


@dataclass(frozen=True, slots=True)
class BoundMapResolver:
    """Scene-compatible resolver bound to one provider/reference pair."""

    provider: MapProvider
    reference: MapReference

    def __call__(self, scene: Scene) -> MapGraph | None:
        """Resolve the map for *scene* using the bound provider and reference."""
        return self.provider.resolve(scene, self.reference)


@dataclass(frozen=True, slots=True)
class SharedMapProvider(MapProvider):
    """Map provider backed by shared-memory map graph names."""

    shared_names: dict[str | None, str] | str
    extractor: MapExtractor | None = None

    @override
    def resolve(self, scene: Scene, reference: MapReference) -> MapGraph | None:
        key = reference.map_key or scene.map_key

        name = (
            self.shared_names.get(key) if isinstance(self.shared_names, dict) else self.shared_names
        )
        if name is None:
            return None

        with MapGraph.from_shared(name) as map_graph:
            if self.extractor is None:
                return map_graph.copy()

            extracted = self.extractor(scene, map_graph)
            return extracted.copy() if extracted is map_graph else extracted
