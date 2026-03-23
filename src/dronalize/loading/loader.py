from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Concatenate, Generic, Protocol

import polars as pl

from dronalize._internal._typing import P, SourceId, SourceT
from dronalize.maps.resolver import MapKey, MapResolver

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dronalize.categories import DatasetSplit
    from dronalize.scene import Scene


MapContext = MapResolver | MapKey | None
IngestOutput = tuple[pl.LazyFrame, MapContext]


@dataclass(slots=True, frozen=True)
class Source(Generic[SourceT]):
    """Represents a raw data source for a scene, identified by a unique identifier."""

    identifier: SourceId
    """Generic identifier for the source, e.g., file name, URL, database key."""
    inner: SourceT
    """The actual source data, which can be of any type (e.g., file path, raw data)."""
    map_key: MapKey = None
    """Optional map key associated with this source."""
    split_assignment: DatasetSplit | None = None
    """Optional concrete dataset split assignment for this source."""
    split_assignment_override: bool = False
    """Whether this split assignment should override BaseSceneLoader inference."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata associated with the source."""

    def with_split_assignment(self, split_assignment: DatasetSplit | None) -> Source[SourceT]:
        """Return a copy with a concrete split assignment."""
        return replace(
            self,
            split_assignment=split_assignment,
            split_assignment_override=False,
        )

    def override_split_assignment(self, split_assignment: DatasetSplit | None) -> Source[SourceT]:
        """Return a copy whose split assignment overrides inferred split routing."""
        return replace(
            self,
            split_assignment=split_assignment,
            split_assignment_override=True,
        )


class SceneLoader(Protocol):
    """Minimal protocol for scene loading, used for type hinting."""

    def scenes(self) -> Iterable[Scene]:
        """Process scenes and yield them one by one.

        This is the main method for processing scenes. It yields `Scene` objects
        one at a time, allowing for memory-efficient processing of large datasets.

        Yields
        ------
        Scene
            Each processed scene, with its identifier and associated data.

        """
        ...

    def scenes_callback(
        self,
        callback: Callable[Concatenate[Scene, P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Process scenes and call the provided callback on each scene.

        This is an alternative to `scenes()` that allows for more flexible
        processing of scenes without needing to yield them. The callback will be
        called with each processed scene, allowing for custom handling (e.g.,
        saving to disk, feeding into a model) without needing to store all scenes
        in memory at once.

        Parameters
        ----------
        callback : Callable
            A function that takes a Scene and additional arguments, and
            processes it (e.g., saves to disk, feeds into a model).
        *args : Any
            Additional positional arguments to pass to the callback.
        **kwargs : Any
            Additional keyword arguments to pass to the callback.

        """
        ...


class ProcessableLoader(Protocol, Generic[SourceT]):
    """Minimal protocol required to work with a loader abstraction.

    This protocol defines the essential interface for discovering data sources,
    tracking dataset sizes, processing raw sources into tabular data, and
    constructing final scene objects.

    """

    def sources(self) -> Iterable[Source[SourceT]]:
        """Discover and yield the data sources to be processed.

        Returns
        -------
        Iterable[Source[SourceT]]
            An iterable containing the raw data sources to process.
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

    def num_scenes(self) -> int | None:
        """Get the total number of scenes that will be generated.

        Returns
        -------
        int or None
            Total number of scenes, or None if the count is unknown or depends
            on dynamic processing (e.g., sliding window extraction).
        """
        ...

    def process_next(self, source: Source[SourceT]) -> Iterable[tuple[pl.DataFrame, MapContext]]:
        """Process a single raw data source into data frames and map contexts.

        Parameters
        ----------
        source : Source[SourceT]
            The raw data source to process.

        Yields
        ------
        tuple[pl.DataFrame, MapContext]
            Processed Polars DataFrames paired with their corresponding map context.
        """
        ...

    def create_scene(
        self,
        df: pl.DataFrame,
        source: Source[SourceT],
        *,
        scene_number: int,
        resolver: MapContext | None = None,
        split: DatasetSplit | None = None,
    ) -> Scene:
        """Construct a Scene object from processed data.

        Parameters
        ----------
        df : pl.DataFrame
            The processed data frame containing the scene data.
        source : Source[SourceT]
            The originating raw data source.
        resolver : MapContext | None, optional
            The map context or resolver associated with the scene.
        scene_number : int | None, optional
            An optional numeric index or identifier for the generated scene.
        split : DatasetSplit | None, optional
            The dataset split (train/val/test) that this scene belongs to.

        Returns
        -------
        Scene
            The fully constructed scene object.
        """
        ...
