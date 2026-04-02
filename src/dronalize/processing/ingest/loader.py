"""Loader-side data structures and protocols for source-to-scene processing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Concatenate, Generic, Protocol

from dronalize._internal.typing import P, SourceId, SourceT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import polars as pl

    from dronalize.core.categories import DatasetSplit
    from dronalize.core.scene import Scene
    from dronalize.processing.maps.resolver import MapKey, MapResolver


@dataclass(slots=True)
class StableSceneIdentifier:
    """Stable identifier for a scene.

    The identifier should remain consistent across runs as long as the raw
    sources and their organization remain unchanged. That makes scene-level and
    source-level split assignment reproducible even if the internal loader
    implementation changes.
    """

    source_identifier: SourceId
    """Stable identifier for the source, e.g., file name, URL, database key."""
    source_local_scene_index: int
    """Zero-based scene index within its source, stable across runs."""


@dataclass(slots=True, frozen=True)
class MapBinding:
    """Loader-side map attachment carried alongside ingested or processed data.

    A binding can provide a stable map key, a resolver, or both. The base
    loader combines this with any source-level map key before constructing the
    final `Scene`.
    """

    map_key: MapKey = None
    """Stable map identifier for the scene, if one is known at ingest time."""
    map_resolver: MapResolver | None = None
    """Resolver that can materialize the map graph for the scene."""


@dataclass(slots=True, frozen=True)
class IngestedData:
    """One source-derived lazy frame plus any scene-level map binding."""

    frame: pl.LazyFrame
    map_binding: MapBinding = field(default_factory=MapBinding)


@dataclass(slots=True, frozen=True)
class ProcessedSceneData:
    """Final scene payload produced immediately before ``Scene`` construction."""

    frame: pl.DataFrame
    stable_identifier: StableSceneIdentifier
    map_binding: MapBinding = field(default_factory=MapBinding)
    predefined_split: DatasetSplit | None = None


@dataclass(slots=True, frozen=True)
class BlockSplitSupport:
    """Loader metadata required to apply block-based split strategies."""

    time_column: str = "frame"
    group_columns: str | tuple[str, ...] | None = None


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
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata associated with the source."""

    def with_predefined_split(self, split_assignment: DatasetSplit | None) -> Source[SourceT]:
        """Return a copy with a concrete split assignment."""
        return replace(self, predefined_split=split_assignment)


class SceneLoader(Protocol):
    """Minimal scene-loading protocol used in public type annotations."""

    def scenes(self) -> Iterable[Scene]:
        """Yield processed scenes one by one."""
        ...

    def for_each_scene(
        self, callback: Callable[Concatenate[Scene, P], None], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        """Call `callback` for each processed scene.

        Parameters
        ----------
        callback : Callable
            Function called with each scene and any extra arguments.
        *args : Any
            Additional positional arguments to pass to the callback.
        **kwargs : Any
            Additional keyword arguments to pass to the callback.
        """
        ...


class ProcessableLoader(Protocol, Generic[SourceT]):
    """Protocol required by the runtime executors and pipeline helpers."""

    def sources(self) -> Iterable[Source[SourceT]]:
        """Discover and yield lightweight sources to be processed.

        Returns
        -------
        Iterable[Source[SourceT]]
            An iterable containing the raw sources to process.
        """
        ...

    def num_sources(self) -> int | None:
        """Get the total number of sources that will be processed.

        Returns
        -------
        int or None
            Total number of sources, or None if the count is unknown or
            expensive to compute in advance.
        """
        ...

    def process_next(self, source: Source[SourceT]) -> Iterable[ProcessedSceneData]:
        """Process one source into finalized scene payloads.

        Parameters
        ----------
        source : Source[SourceT]
            The source to process.

        Yields
        ------
        ProcessedSceneData
            Processed scene payloads with their split metadata resolved.
        """
        ...

    def create_scene(
        self, data: ProcessedSceneData, source: Source[SourceT], scene_number: int
    ) -> Scene:
        """Construct a Scene object from processed data.

        Parameters
        ----------
        data : ProcessedSceneData
            Processed scene payload.
        source : Source[SourceT]
            Originating source.
        scene_number : int
            Monotonic scene number assigned by the loader.

        Returns
        -------
        Scene
            Fully constructed scene object.
        """
        ...
