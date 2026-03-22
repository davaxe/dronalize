from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Concatenate

from typing_extensions import override

from dronalize._internal._typing import P, SourceT
from dronalize.categories import DatasetSplit
from dronalize.config.map import MapConfig
from dronalize.exceptions import SplitNotSupportedError
from dronalize.loading.loader import (
    IngestOutput,
    MapContext,
    ProcessableLoader,
    SceneLoader,
    Source,
)
from dronalize.maps.resolver import MapKey, MapResolver, no_map
from dronalize.scene import CANONICAL_V1, Scene, SceneField, SceneSchema

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import polars as pl

    from dronalize.config.loader import LoaderConfig
    from dronalize.pipeline.pipeline import Pipeline


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

    @property
    def requested_scene_schema(self) -> SceneSchema | None:
        """Return the schema requested by downstream consumers, if any."""
        return self._output_schema

    @property
    def output_scene_schema(self) -> SceneSchema:
        """Return the effective scene schema this loader should optimize for."""
        return self.native_scene_schema() if self._output_schema is None else self._output_schema

    def set_output_schema(self, schema: SceneSchema | None) -> None:
        """Update the schema requested by downstream consumers."""
        self._output_schema = schema

    def requires_scene_fields(self, *fields: SceneField | str) -> bool:
        """Return whether the effective output schema needs all requested fields."""
        return self.output_scene_schema.has(*fields)

    @classmethod
    @abstractmethod
    def default_config(cls) -> LoaderConfig:
        """Return the default loader configuration for this dataset."""

    @classmethod
    @abstractmethod
    def native_scene_schema(cls) -> SceneSchema:
        """Return the schema produced by this loader before output conversion."""

    @classmethod
    def predefined_splits(cls) -> tuple[DatasetSplit, ...]:
        """Return predefined dataset partitions supported by this loader."""
        return ()

    @classmethod
    def default_map_config(cls) -> MapConfig:
        """Return the default map configuration for this dataset."""
        return MapConfig.default()

    @abstractmethod
    def ingest(self, source: Source[SourceT]) -> Iterable[IngestOutput]:
        """Read a raw data source into one or more `(LazyFrame, MapContext)` pairs."""

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
        supported_splits = self.predefined_splits()
        if len(supported_splits) == 0:
            yield from self._assign_splits(self.discover_sources(), None)
            return

        for split in supported_splits:
            yield from self._assign_splits(self.sources_for_split(split), split)

    def map_resolver(self) -> MapResolver:  # noqa: PLR6301 (will be overridden)
        """Return a default resolver for this dataset's map data."""
        return no_map()

    @override
    def sources(self) -> Iterable[Source[SourceT]]:
        """Return sources for the currently configured split(s)."""
        if self.splits is None:
            yield from self.all_sources()
            return

        supported_splits = self.predefined_splits()
        if len(supported_splits) == 0:
            raise SplitNotSupportedError(type(self).__name__, self.splits)

        for split in self.splits:
            if split not in supported_splits:
                raise SplitNotSupportedError(type(self).__name__, split)
            yield from self._assign_splits(self.sources_for_split(split), split)

    @staticmethod
    def _assign_splits(
        sources: Iterable[Source[SourceT]],
        split: DatasetSplit | None,
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
            for scene_df, map_context in self.process_next(source):
                yield self.create_scene(
                    scene_df,
                    source,
                    scene_number=self._scene_counter,
                    map_context=map_context,
                )
                self._scene_counter += 1

    @override
    def create_scene(
        self,
        df: pl.DataFrame,
        source: Source[SourceT],
        *,
        scene_number: int,
        map_context: MapContext | None = None,
        split: DatasetSplit | None = None,
    ) -> Scene:
        """Create a Scene object from the processed DataFrame and its source."""
        map_key = source.map_key or (map_context if isinstance(map_context, str) else None)
        if not self.map_config.include_map:
            map_resolver = None
            map_key = None
        else:
            map_resolver = map_context if not isinstance(map_context, str) else self.map_resolver()
        scene = Scene(
            inner=df,
            number=scene_number,
            input_len=self.loader_config.resampled_input_len,
            output_len=self.loader_config.resampled_output_len,
            schema=self.native_scene_schema(),
            sample_time=self.loader_config.post_sample_time,
            map_key=map_key,
            map_resolver=map_resolver,
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

        for raw_lazy_frame, map_context in self.ingest(source):
            for df in self._pipeline.execute(raw_lazy_frame, collect=True, filter_empty=True):
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
