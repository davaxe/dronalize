from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Concatenate, Generic, Protocol

import polars as pl
from typing_extensions import override

from dronalize._internal._types import P, SourceId, SourceT
from dronalize.categories import DatasetSplit
from dronalize.config.map import MapConfig
from dronalize.exceptions import SplitNotSupportedError
from dronalize.maps.resolver import MapKey, MapResolver, no_map
from dronalize.scene import CANONICAL_V1, Scene, SceneSchema

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dronalize.config.loader import LoaderConfig
    from dronalize.pipeline.pipeline import Pipeline


MapContext = MapResolver | MapKey | None
IngestOutput = tuple[pl.LazyFrame, MapContext]


# TODO: Maybe remove metadata
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


class BaseSceneLoader(ABC, SceneLoader, ProcessableLoader[SourceT]):
    """ABC interface for processing raw data sources into a standardized format."""

    _shared_memory_name: ClassVar[dict[MapKey, str] | str | None] = None

    def __init__(
        self,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        *,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        output_schema: SceneSchema | None = CANONICAL_V1,
    ) -> None:
        self._output_schema: SceneSchema | None = output_schema
        self._pipeline: Pipeline | None = None
        self.loader_config: LoaderConfig = loader_config or self.default_config()
        self.map_config: MapConfig = map_config or self.default_map_config()
        self._scene_counter: int = 0

        if splits is None:
            self.splits: list[DatasetSplit] | None = None
        elif isinstance(splits, DatasetSplit):
            self.splits = [splits]
        else:
            self.splits = list(splits)

    @classmethod
    @abstractmethod
    def default_config(cls) -> LoaderConfig:
        """Return the default loader configuration for this dataset."""

    @classmethod
    @abstractmethod
    def native_scene_schema(cls) -> SceneSchema:
        """Return the schema produced by this loader before output conversion."""
        return CANONICAL_V1

    @classmethod
    def default_map_config(cls) -> MapConfig:
        """Return the default map configuration for this dataset."""
        return MapConfig.default()

    @abstractmethod
    def ingest(self, source: Source[SourceT]) -> Iterable[IngestOutput]:
        """Read a raw data source into one or more `(LazyFrame, MapResolver)` pairs."""

    @abstractmethod
    def pipeline(self) -> Pipeline:
        """Return the composable processing pipeline for this loader."""

    def discover_sources(self) -> Iterable[Source[SourceT]]:
        """Discover sources for datasets that do not define predefined splits."""
        msg = f"{type(self).__name__} must implement discover_sources() or sources_for_split()."
        raise NotImplementedError(msg)

    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[SourceT]]:
        """Return sources belonging to a specific predefined split.

        Override this method in subclasses whose underlying dataset provides predefined splits.
        """
        raise SplitNotSupportedError(type(self).__name__, split)

    def all_sources(self) -> Iterable[Source[SourceT]]:
        """Discover and yield sources across all available dataset partitions."""
        found_any_split = False

        for split in DatasetSplit:
            try:
                yield from self._assign_splits(self.sources_for_split(split), split)
                found_any_split = True
            except SplitNotSupportedError:  # noqa: PERF203
                continue

        if not found_any_split:
            yield from self._assign_splits(self.discover_sources(), None)

    def map_resolver(self) -> MapResolver:  # noqa: PLR6301 (will be overriden)
        """Return a default resolver for this dataset's map data."""
        return no_map()

    @override
    def sources(self) -> Iterable[Source[SourceT]]:
        """Return sources for the currently configured split(s)."""
        if self.splits is None:
            yield from self.all_sources()
            return

        for split in self.splits:
            yield from self._assign_splits(self.sources_for_split(split), split)

    @staticmethod
    def _assign_splits(
        sources: Iterable[Source[SourceT]], split: DatasetSplit | None
    ) -> Iterable[Source[SourceT]]:
        """Apply effective split metadata to yielded sources."""
        for source in sources:
            target = (
                source.split_assignment
                if source.split_assignment_override
                else (source.split_assignment or split)
            )
            yield (
                source
                if source.split_assignment == target
                else source.with_split_assignment(target)
            )

    @override
    def scenes_callback(
        self,
        callback: Callable[Concatenate[Scene, P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        for scene in self.scenes():
            callback(scene, *args, **kwargs)

    @override
    def scenes(self) -> Iterable[Scene]:
        for source in self.sources():
            for scene_df, resolver in self.process_next(source):
                yield self.create_scene(
                    scene_df, source, scene_number=self._scene_counter, resolver=resolver
                )
                self._scene_counter += 1

    @override
    def create_scene(
        self,
        df: pl.DataFrame,
        source: Source[SourceT],
        *,
        scene_number: int,
        resolver: MapContext | None = None,
        split: DatasetSplit | None = None,
    ) -> Scene:
        """Create a Scene object from the processed DataFrame and its source."""
        map_key = source.map_key or (resolver if isinstance(resolver, str) else None)
        resolver = resolver if not isinstance(resolver, str) else self.map_resolver()
        scene = Scene(
            inner=df,
            number=number,
            input_len=self.loader_config.resampled_input_len,
            output_len=self.loader_config.resampled_output_len,
            schema=self.native_scene_schema(),
            sample_time=self.loader_config.post_sample_time,
            map_key=map_key,
            map_resolver=resolver,
            split_assignment=split if split is not None else source.split_assignment,
        )
        return scene if self._output_schema is None else scene.as_schema(self._output_schema)

    @override
    def process_next(
        self,
        source: Source[SourceT],
    ) -> Iterable[tuple[pl.DataFrame, MapContext]]:
        if self._pipeline is None:
            self._pipeline = self.pipeline()

        for raw_lf, map_context in self.ingest(source):
            for df in self._pipeline.execute(raw_lf, collect=True, filter_empty=True):
                yield df, map_context

    @override
    def num_scenes(self) -> int | None:
        return None

    @override
    def num_sources(self) -> int | None:
        return None

    @classmethod
    def set_shared_memory(
        cls,
        name: str | None = None,
        mappings: dict[MapKey, str] | None = None,
    ) -> None:
        """Set the shared memory name or mapping for this loader class."""
        if mappings is not None and name is not None:
            msg = "Cannot set both 'name' and 'mappings'. Please choose one."
            raise ValueError(msg)

        if mappings is not None:
            cls._shared_memory_name = mappings
        elif name is not None:
            cls._shared_memory_name = name
        else:
            cls._shared_memory_name = None
