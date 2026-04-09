"""Base scene-loader implementation and split-assignment helpers."""

from __future__ import annotations

import itertools
import math
import operator
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Concatenate, Generic, cast

from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import TypeVar

from dronalize._internal.typing import P, SourceT
from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import (
    LoaderConfigError,
    SplitAssignmentError,
    SplitNotSupportedError,
    SplitStrategyNotSupportedError,
)
from dronalize.core.scene import Scene, TrajectoryField, TrajectorySchema
from dronalize.processing.loading.assigner import SplitAssigner, StatelessWeightedAssigner
from dronalize.processing.loading.loader import (
    LoadedSourceData,
    MapBinding,
    PreparedSceneData,
    SceneIdentifier,
    Source,
)
from dronalize.processing.loading.splits import (
    NativeSplitStrategy,
    NoSplitStrategy,
    SceneSplitStrategy,
    ShuffledTimeBlockStrategy,
    SourceSplitStrategy,
    TimeBlockStrategy,
)
from dronalize.processing.maps.resolver import MapKey, MapResolver, no_map
from dronalize.processing.pipeline.factory import trajectory_pipeline
from dronalize.processing.pipeline.presets import (
    lane_change_sampling_spec,
    standard_trajectory_spec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from dronalize.processing.loading.config import LoaderConfig
    from dronalize.processing.loading.splits import SplitConfig, SplitStrategyName
    from dronalize.processing.maps.config import MapConfig
    from dronalize.processing.pipeline.pipeline import Pipeline


class LoaderOptions(BaseModel):
    """Base model for dataset-specific loader options.

    Loaders that need extra configuration beyond the shared `LoaderConfig`
    should expose a typed subclass through `loader_options_model()`. This keeps
    the public dataset descriptor precise while still allowing dataset-specific
    configuration to be validated and documented.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class NoLoaderOptions(LoaderOptions):
    """Empty loader options for datasets that do not require any configuration."""


@dataclass(slots=True, frozen=True, kw_only=True)
class LoaderSplitCapabilities:
    """Advertised custom split capabilities for a loader class.

    These flags do not implement splitting on their own. They define which
    loader-side split strategies should be exposed in metadata and accepted during
    split-request validation.
    """

    supports_source_split: bool = False
    """Enable source-level assignment for datasets with many independent sources."""
    supports_scene_split: bool = False
    """Enable scene-level assignment when a source already contains many scenes."""
    supports_block_split: bool = False
    """Enable block-based assignment for long recordings that stay source-local."""


_LoaderOptionsT = TypeVar("_LoaderOptionsT", bound=LoaderOptions, default=NoLoaderOptions)


class BaseSceneLoader(ABC, Generic[SourceT, _LoaderOptionsT]):
    """Base class for turning raw dataset sources into standardized scenes.

    This is the main extension hook for custom dataset integrations. Subclasses
    own dataset-specific concerns such as discovering sources, parsing raw
    records, normalizing coordinates, and attaching map context. Shared runtime
    concerns such as filtering, resampling, split handling, and export are
    handled by the surrounding processing pipeline once the loader produces
    normalized scene data.

    Most custom dataset integrations only need to implement:

    - `default_config()`
    - `native_trajectory_schema()`
    - `discover_sources()`
    - `load_source()`
    - `prepare_scene_data()`

    Datasets can optionally override the split metadata, map defaults, and
    loader-options model when they need richer behavior.
    """

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities()
    _shared_memory_name: ClassVar[dict[MapKey, str] | str | None] = None

    def __init__(
        self,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
        trajectory_schema: TrajectorySchema | None = None,
        loader_options: _LoaderOptionsT | None = None,
    ) -> None:
        """Initialize the loader with the given configuration and split request.

        Parameters
        ----------
        loader_config : LoaderConfig | None, optional
            Configuration for the loader's processing pipeline. When omitted,
            the loader uses its default configuration.
        map_config : MapConfig | None, optional
            Configuration for map data processing. `None` indicates that
            the loader should not attempt to resolve any map data for its scenes.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Predefined dataset splits to load. When omitted, the loader yields
            sources across all available splits. When specified, the loader
            validates that the requested splits are supported and only yields
            sources belonging to those
            splits.
        split_request : SplitConfig | None, optional
            Custom split request for dynamic assignment. When omitted, the
            loader does not apply any custom split strategy and only uses
            predefined splits if applicable.
        trajectory_schema : TrajectorySchema | None, optional
            Schema to convert output scenes to before yielding. When omitted,
            the loader produces scenes in its native schema. When explicitly set
            to `None`, the loader also produces scenes in its native schema but
            downstream consumers are not informed of this fact through metadata.
        loader_options : LoaderOptions | None, optional
            Dataset-specific options validated against this loader's
            `loader_options_model()`. When omitted, the loader uses
            `default_loader_options()`.

        """
        self.loader_options: _LoaderOptionsT = loader_options or self.default_loader_options()
        self._trajectory_schema: TrajectorySchema | None = trajectory_schema
        self._pipeline: Pipeline | None = None
        self.split_request: SplitConfig | None = split_request
        self.loader_config: LoaderConfig = loader_config or self.default_config()
        self.map_config: MapConfig | None = map_config
        self._scene_counter: int = 0
        self._rng: random.Random = (
            random.Random(self.split_request.seed)
            if self.split_request is not None
            else random.Random()
        )

        self._scene_assigner: SplitAssigner | None = None
        if split_request is not None and isinstance(split_request.strategy, SceneSplitStrategy):
            self._scene_assigner = StatelessWeightedAssigner(
                split_request.active_splits(),
                split_request.active_weights(),
                seed=split_request.seed,
            )

        if splits is None:
            self.splits: list[DatasetSplit] | None = None
        elif isinstance(splits, DatasetSplit):
            self.splits = [splits]
        else:
            self.splits = list(splits)

        self._validate_split_request(split_request)

    @property
    def requested_trajectory_schema(self) -> TrajectorySchema | None:
        """Return the schema requested by downstream consumers, if any."""
        return self._trajectory_schema

    @property
    def effective_trajectory_schema(self) -> TrajectorySchema:
        """Return the effective trajectory schema this loader should optimize for."""
        return (
            self.native_trajectory_schema()
            if self._trajectory_schema is None
            else self._trajectory_schema
        )

    def set_trajectory_schema(self, schema: TrajectorySchema | None) -> None:
        """Update the schema requested by downstream consumers."""
        self._trajectory_schema = schema

    def requires_trajectory_fields(self, *fields: TrajectoryField | str) -> bool:
        """Return whether the effective output schema needs all requested fields."""
        return self.effective_trajectory_schema.has(*fields)

    @classmethod
    def _validate_split_request(cls, split_request: SplitConfig | None) -> None:
        """Reject direct split requests that this loader does not advertise."""
        if split_request is None:
            return
        if isinstance(split_request.strategy, (NoSplitStrategy, NativeSplitStrategy)):
            return

        supported_strategies = cls.supported_split_strategies()
        if split_request.strategy_name not in supported_strategies:
            raise SplitStrategyNotSupportedError(
                cls.__name__, split_request.strategy_name, supported_strategies
            )

    @classmethod
    @abstractmethod
    def default_config(cls) -> LoaderConfig:
        """Return the default loader configuration for this dataset."""

    @classmethod
    @abstractmethod
    def native_trajectory_schema(cls) -> TrajectorySchema:
        """Return the schema produced by this loader before output conversion."""

    @classmethod
    def predefined_splits(cls) -> tuple[DatasetSplit, ...]:
        """Return predefined dataset partitions supported by this loader."""
        return ()

    @classmethod
    def supported_split_strategies(cls) -> tuple[SplitStrategyName, ...]:
        """Return custom split strategies implemented by this loader.

        The default implementation derives the list from `cls.split_capabilities`, but
        subclasses can override it when they need more control.
        """
        return (
            *(("source",) if cls.split_capabilities.supports_source_split else ()),
            *(("scene",) if cls.split_capabilities.supports_scene_split else ()),
            *(("time", "shuffled-time") if cls.split_capabilities.supports_block_split else ()),
        )

    @classmethod
    def recommended_split_strategy(cls) -> SplitStrategyName | None:
        """Return the preferred custom split strategy for automatic selection, if any."""
        strategies = cls.supported_split_strategies()
        return strategies[0] if len(strategies) == 1 else None

    @classmethod
    def default_map_config(cls) -> MapConfig | None:
        """Return the default map configuration for this dataset."""
        return None

    @classmethod
    def loader_options_model(cls) -> type[_LoaderOptionsT]:
        """Return the typed model used for validating dataset-specific options."""
        # The cast keeps the default case lightweight for loaders without
        # dataset-specific options.
        return cast("type[_LoaderOptionsT]", NoLoaderOptions)

    @classmethod
    def default_loader_options(cls) -> _LoaderOptionsT:
        """Return the default dataset-specific options instance for this loader."""
        return cls.loader_options_model()()

    @classmethod
    def parse_loader_options(cls, options_dict: Mapping[str, object] | None) -> _LoaderOptionsT:
        """Parse a dictionary of configuration options into this loader's options model."""
        try:
            return cls.loader_options_model().model_validate(options_dict or {})
        except ValidationError as e:
            msg = f"Invalid loader options for {cls.__name__}: {e}"
            raise LoaderConfigError(msg) from e

    @abstractmethod
    def load_source(self, source: Source[SourceT]) -> Iterable[LoadedSourceData]:
        """Read one source into lazy frames plus any scene-level map bindings."""

    def pipeline(self) -> Pipeline:
        """Return the composable processing pipeline for this loader.

        By default, loaders use the shared trajectory-processing pipeline and
        pass through any active custom split request. Subclasses can still
        override this when they need dataset-specific post-processing.
        """
        return trajectory_pipeline(
            standard_trajectory_spec(self.loader_config, split_request=self.split_request)
            if self.loader_config.lane_change_sampling is None
            else lane_change_sampling_spec(self.loader_config, split_request=self.split_request)
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
            yield from self._assign_source_splits(
                zip(self.discover_sources(), itertools.repeat(None))
            )
            return

        yield from self._assign_source_splits(
            itertools.chain.from_iterable(
                zip(self.sources_for_split(split), itertools.repeat(split))
                for split in supported_splits
            )
        )

    def map_resolver(self) -> MapResolver:
        """Return a default resolver for this dataset's map data."""
        _ = self
        return no_map()

    def _assign_source_splits(
        self, sources: Iterable[tuple[Source[SourceT], DatasetSplit | None]]
    ) -> Iterable[Source[SourceT]]:
        """Assign splits to sources according to the active split request, if any."""
        if self.split_request is not None and isinstance(
            self.split_request.strategy, SourceSplitStrategy
        ):
            # If `SourceSplitStrategy` is requested, assign splits at the source level
            sources = self._by_source_assignments(
                [source for source, _ in sources], self.split_request
            )

        for source, split in sources:
            yield source.with_predefined_split(split)

    def _by_source_assignments(
        self, sources: list[Source[SourceT]], request: SplitConfig
    ) -> list[tuple[Source[SourceT], DatasetSplit]]:
        self._rng.shuffle(sources)
        n_sources = len(sources)
        active_splits = request.active()
        raw_counts = {split: n_sources * weight for split, weight in active_splits}
        counts = {split: math.floor(raw) for split, raw in raw_counts.items()}
        remaining = n_sources - sum(counts.values())
        remainders = sorted(
            ((split, raw_counts[split] - counts[split]) for split in counts),
            key=operator.itemgetter(1),
            reverse=True,
        )
        for split, _ in remainders[:remaining]:
            counts[split] += 1

        split_labels = [split for split, count in counts.items() for _ in range(count)]
        return list(zip(sources, split_labels, strict=True))

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
            yield from self._assign_source_splits(
                zip(self.sources_for_split(split), itertools.repeat(split))
            )

    def for_each_scene(
        self, callback: Callable[Concatenate[Scene, P], None], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        """Call `callback` for each scene produced by :meth:`scenes`."""
        for scene in self.scenes():
            callback(scene, *args, **kwargs)

    def scenes(self) -> Iterable[Scene]:
        """Yield fully constructed scenes across all selected sources."""
        for source in self.sources():
            for processed in self.process_next(source):
                yield self.create_scene(processed, source, self._scene_counter)
                self._scene_counter += 1

    def create_scene(
        self, data: PreparedSceneData, source: Source[SourceT], scene_number: int
    ) -> Scene:
        """Create a Scene object from the processed DataFrame and its source."""
        map_key, map_resolver = self._resolve_scene_map(source, data.map_binding)
        scene = Scene(
            frame=data.frame,
            scene_number=scene_number,
            input_len=self.loader_config.resampled_input_len,
            output_len=self.loader_config.resampled_output_len,
            schema=self.native_trajectory_schema(),
            sample_time=self.loader_config.post_sample_time,
            map_key=map_key,
            map_resolver=map_resolver,
            split_assignment=self._resolve_scene_split_assignment(data, source),
        )
        return (
            scene if self._trajectory_schema is None else scene.as_schema(self._trajectory_schema)
        )

    def process_next(self, source: Source[SourceT]) -> Iterable[PreparedSceneData]:
        """Process one source into scene payloads ready for `Scene` creation."""
        if self._pipeline is None:
            self._pipeline = self.pipeline()

        source_local_scene_index = 0
        for data in self.load_source(source):
            for df in self._pipeline.execute(data.frame, collect=True, filter_empty=True):
                yield PreparedSceneData(
                    df,
                    map_binding=data.map_binding,
                    predefined_split=source.predefined_split,
                    stable_identifier=SceneIdentifier(
                        source_identifier=source.identifier,
                        source_local_scene_index=source_local_scene_index,
                    ),
                )
                source_local_scene_index += 1

    def num_sources(self) -> int | None:
        """Return the number of sources when known in advance."""
        _ = self
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
        self, data: PreparedSceneData, request: SplitConfig
    ) -> DatasetSplit | None:
        identifier = data.stable_identifier
        match request.strategy:
            case SceneSplitStrategy():
                return (
                    self._scene_assigner.assign(
                        identifier.source_local_scene_index, identifier.source_identifier
                    )
                    if self._scene_assigner is not None
                    else None
                )
            case TimeBlockStrategy() | ShuffledTimeBlockStrategy():
                if "split" in data.frame:
                    split_str = str(data.frame["split"].first())
                    try:
                        return DatasetSplit(split_str.lower())
                    except ValueError as exc:
                        msg = f"Invalid split assignment '{split_str}' in dataframe."
                        raise SplitAssignmentError(msg) from exc
                msg = "Did not get split column in dataframe"
                raise SplitAssignmentError(msg)
            case NoSplitStrategy() | NativeSplitStrategy() | SourceSplitStrategy():
                msg = "These cases should have been handled before calling this method."
                raise RuntimeError(msg)

    def _resolve_scene_split_assignment(
        self, data: PreparedSceneData, source: Source[SourceT]
    ) -> DatasetSplit | None:
        """Resolve the effective split assignment for one created scene."""
        request = self.split_request
        if request is None:
            return source.predefined_split
        if isinstance(request.strategy, NoSplitStrategy):
            return None
        if isinstance(request.strategy, NativeSplitStrategy | SourceSplitStrategy):
            return source.predefined_split
        return self._resolve_split_assignment(data, request)

    def _resolve_scene_map(
        self, source: Source[SourceT], map_binding: MapBinding
    ) -> tuple[MapKey, MapResolver | None]:
        """Resolve the effective map key and resolver for one processed scene."""
        map_key = map_binding.map_key if map_binding.map_key is not None else source.map_key

        if self.map_config is None:
            return None, None

        map_resolver = map_binding.map_resolver
        if map_resolver is None and map_key is not None:
            map_resolver = self.map_resolver()

        return map_key, map_resolver
