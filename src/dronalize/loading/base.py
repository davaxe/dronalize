from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Concatenate

from typing_extensions import override

from dronalize._internal._typing import P, SourceT
from dronalize.categories import DatasetSplit
from dronalize.config.map import MapConfig
from dronalize.config.split import (
    BySceneSplit,
    BySourceSplit,
    NativeSplit,
    ShuffledTimeBlockSplit,
    TimeBlockSplit,
    Unsplit,
)
from dronalize.exceptions import SplitMethodNotSupportedError, SplitNotSupportedError
from dronalize.loading.assigner import SplitAssigner, StatelessWeightedAssigner
from dronalize.loading.loader import (
    IngestedData,
    MapBinding,
    ProcessableLoader,
    ProcessedSceneData,
    SceneLoader,
    Source,
    StableSceneIdentifier,
)
from dronalize.maps.resolver import MapKey, MapResolver, no_map
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.scene import CANONICAL_V1, Scene, SceneField, SceneSchema

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dronalize.config.loader import LoaderConfig
    from dronalize.config.split import SplitRequest, SplitStrategyName
    from dronalize.pipeline.pipeline import Pipeline


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseSceneLoaderConfig:
    """Advertised custom split capabilities for a loader class.

    These flags do not implement splitting on their own. They define which
    loader-side split methods should be exposed in metadata and accepted during
    split-request validation.
    """

    source_split_enabled: bool = False
    """Enable source-level assignment for datasets with many independent sources."""
    scene_split_enabled: bool = False
    """Enable scene-level assignment when a source already contains many scenes."""
    block_split_enabled: bool = False
    """Enable block-based assignment for long recordings that stay source-local."""


class BaseSceneLoader(ABC, SceneLoader, ProcessableLoader[SourceT]):
    """Base class for turning lightweight sources into standardized scenes."""

    config: ClassVar[BaseSceneLoaderConfig] = BaseSceneLoaderConfig()
    _shared_memory_name: ClassVar[dict[MapKey, str] | str | None] = None

    def __init__(
        self,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        *,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitRequest | None = None,
        output_schema: SceneSchema | None = CANONICAL_V1,
    ) -> None:
        self._output_schema: SceneSchema | None = output_schema
        self._pipeline: Pipeline | None = None
        self.split_request: SplitRequest | None = split_request
        self.loader_config: LoaderConfig = loader_config or self.default_config()
        self.map_config: MapConfig = map_config or self.default_map_config()
        self._scene_counter: int = 0
        self._assigners: dict[SplitStrategyName, SplitAssigner] = {}

        if splits is None:
            self.splits: list[DatasetSplit] | None = None
        elif isinstance(splits, DatasetSplit):
            self.splits = [splits]
        else:
            self.splits = list(splits)

        self._validate_split_request(split_request)

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
    def _validate_split_request(cls, split_request: SplitRequest | None) -> None:
        """Reject direct split requests that this loader does not advertise."""
        if split_request is None:
            return
        if isinstance(split_request.strategy, (Unsplit, NativeSplit)):
            return

        supported_methods = cls.supported_split_methods()
        if split_request.method not in supported_methods:
            raise SplitMethodNotSupportedError(
                cls.__name__,
                split_request.method,
                supported_methods,
            )

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
    def supported_split_methods(cls) -> tuple[SplitStrategyName, ...]:
        """Return custom split methods implemented by this loader.

        The default implementation derives the list from `cls.config`, but
        subclasses can override it when they need more control.
        """
        return (
            *(("by_source",) if cls.config.source_split_enabled else ()),
            *(("by_scene",) if cls.config.scene_split_enabled else ()),
            *(("time_blocks", "shuffled_time_blocks") if cls.config.block_split_enabled else ()),
        )

    @classmethod
    def recommended_split_method(cls) -> SplitStrategyName | None:
        """Return the preferred custom split method for automatic selection, if any."""
        methods = cls.supported_split_methods()
        return methods[0] if len(methods) == 1 else None

    @classmethod
    def default_map_config(cls) -> MapConfig:
        """Return the default map configuration for this dataset."""
        return MapConfig.default()

    @abstractmethod
    def ingest(self, source: Source[SourceT]) -> Iterable[IngestedData]:
        """Read one source into lazy frames plus any scene-level map bindings."""

    def pipeline(self) -> Pipeline:
        """Return the composable processing pipeline for this loader.

        By default, loaders use the shared trajectory-processing pipeline and
        pass through any active custom split request. Subclasses can still
        override this when they need dataset-specific post-processing.
        """
        return trajectory_pipeline(
            self.loader_config,
            split_request=self.split_request,
        )

    def discover_sources(self) -> Iterable[Source[SourceT]]:
        """Discover sources for datasets that do not define predefined splits."""
        msg = f"{type(self).__name__} must implement discover_sources() or sources_for_split()."
        raise NotImplementedError(msg)

    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[SourceT]]:
        """Return sources belonging to a specific predefined split.

        Override this in loaders whose raw dataset already defines partitions
        such as train/val/test directories.
        """
        raise SplitNotSupportedError(type(self).__name__, split)

    def all_sources(self) -> Iterable[Source[SourceT]]:
        """Discover and yield sources across all available dataset partitions."""
        supported_splits = self.predefined_splits()
        if len(supported_splits) == 0:
            yield from self._assign_source_splits(self.discover_sources(), None)
            return

        for split in supported_splits:
            yield from self._assign_source_splits(self.sources_for_split(split), split)

    def map_resolver(self) -> MapResolver:
        """Return a default resolver for this dataset's map data."""
        _ = self
        return no_map()

    @staticmethod
    def _assign_source_splits(
        sources: Iterable[Source[SourceT]],
        split: DatasetSplit | None,
    ) -> Iterable[Source[SourceT]]:
        """Apply effective split metadata to yielded sources."""
        for source in sources:
            yield source.with_predefined_split(split)

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
            yield from self._assign_source_splits(self.sources_for_split(split), split)

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
            for processed in self.process_next(source):
                yield self.create_scene(processed, source, self._scene_counter)
                self._scene_counter += 1

    @override
    def create_scene(
        self,
        data: ProcessedSceneData,
        source: Source[SourceT],
        scene_number: int,
    ) -> Scene:
        """Create a Scene object from the processed DataFrame and its source."""
        map_key, map_resolver = self._resolve_scene_map(source, data.map_binding)

        scene = Scene(
            inner=data.frame,
            number=scene_number,
            input_len=self.loader_config.resampled_input_len,
            output_len=self.loader_config.resampled_output_len,
            schema=self.native_scene_schema(),
            sample_time=self.loader_config.post_sample_time,
            map_key=map_key,
            map_resolver=map_resolver,
            split_assignment=self._resolve_scene_split_assignment(data, source),
        )
        return scene if self._output_schema is None else scene.as_schema(self._output_schema)

    @override
    def process_next(self, source: Source[SourceT]) -> Iterable[ProcessedSceneData]:
        if self._pipeline is None:
            self._pipeline = self.pipeline()

        local_scene_number = 0
        for data in self.ingest(source):
            for df in self._pipeline.execute(data.frame, collect=True, filter_empty=True):
                yield ProcessedSceneData(
                    df,
                    map_binding=data.map_binding,
                    predefined_split=source.predefined_split,
                    stable_identifier=StableSceneIdentifier(
                        source_identifier=source.identifier,
                        local_scene_number=local_scene_number,
                    ),
                )
                local_scene_number += 1

    @override
    def num_scenes(self) -> int | None:
        return None

    @override
    def num_sources(self) -> int | None:
        return None

    @classmethod
    def set_shared_memory(
        cls, name: str | None = None, mappings: dict[MapKey, str] | None = None
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

    def _resolve_split_assignment(
        self,
        data: ProcessedSceneData,
        request: SplitRequest,
    ) -> DatasetSplit | None:
        identifier = data.stable_identifier
        match request.strategy:
            case BySourceSplit():
                return self._assigners.setdefault(
                    request.method,
                    StatelessWeightedAssigner(
                        request.active_splits(), request.active_weights(), seed=request.seed
                    ),
                ).assign(identifier.source_identifier)
            case BySceneSplit():
                return self._assigners.setdefault(
                    request.method,
                    StatelessWeightedAssigner(
                        request.active_splits(), request.active_weights(), seed=request.seed
                    ),
                ).assign(
                    identifier.local_scene_number,
                    identifier.source_identifier,
                )
            case TimeBlockSplit() | ShuffledTimeBlockSplit():
                if "split" in data.frame:
                    split_str: str = str(data.frame["split"].first())
                    return DatasetSplit(split_str.lower())
                msg = "Did not get split column in dataframe"
                raise ValueError(msg)
            case Unsplit() | NativeSplit():
                return None

    def _resolve_scene_split_assignment(
        self,
        data: ProcessedSceneData,
        source: Source[SourceT],
    ) -> DatasetSplit | None:
        """Resolve the effective split assignment for one created scene."""
        request = self.split_request
        if request is None:
            return source.predefined_split
        if isinstance(request.strategy, Unsplit):
            return None
        if isinstance(request.strategy, NativeSplit):
            return source.predefined_split
        return self._resolve_split_assignment(data, request)

    def _resolve_scene_map(
        self,
        source: Source[SourceT],
        map_binding: MapBinding,
    ) -> tuple[MapKey, MapResolver | None]:
        """Resolve the effective map key and resolver for one processed scene."""
        map_key = map_binding.key if map_binding.key is not None else source.map_key

        if not self.map_config.include_map:
            return None, None

        map_resolver = map_binding.resolver
        if map_resolver is None and map_key is not None:
            map_resolver = self.map_resolver()

        return map_key, map_resolver
