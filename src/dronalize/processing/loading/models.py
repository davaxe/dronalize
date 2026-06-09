"""Loader-side data structures for DatasetSource-to-scene processing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Generic

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from dronalize.core.typing import SourceId, SourceT
from dronalize.processing.maps import MapReference

if TYPE_CHECKING:
    import polars as pl

    from dronalize.core.categories import DatasetSplit


@dataclass(slots=True, frozen=True)
class LoadedSourceFrame:
    """One DatasetSource-derived lazy frame plus any scene-level map reference."""

    frame: pl.LazyFrame
    map_reference: MapReference = field(default_factory=MapReference)
    source_split: DatasetSplit | None = None


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
