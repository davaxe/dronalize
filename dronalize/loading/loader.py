from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Concatenate, Generic, Protocol

import polars as pl
from typing_extensions import override

from dronalize._internal._types import P, SourceId, SourceT
from dronalize.categories import DatasetSplit
from dronalize.config.map import MapConfig
from dronalize.exceptions import LoaderConfigError, SplitNotSupportedError
from dronalize.maps.resolver import MapKey, MapResolver, no_map
from dronalize.scene import Scene

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dronalize.config.loader import LoaderConfig
    from dronalize.pipeline.pipeline import Pipeline


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
        resolver: MapContext | None = None,
        scene_number: int | None = None,
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
    """ABC interface for processing raw data sources into a standardized format.

    This class contains logic for orchestrating the loading and processing of
    raw data sources into scenes, while allowing subclasses to implement the
    specific details of how raw data is read and processed. The main processing
    flow is as follows:

    1. **Source discovery** — `sources()` resolves the configured split
       selection:

       - `splits=None` → all available sources
       - `DatasetSplit.TRAIN` → `train_sources()`
       - `DatasetSplit.VAL` → `validate_sources()`
       - `DatasetSplit.TEST` → `test_sources()`

       Datasets without predefined splits implement `discover_sources()`.
       Datasets with predefined splits override the relevant split-specific
       source methods. When no split is requested, `BaseSceneLoader` yields all
       available sources without assigning predefined split metadata. When
       explicit splits are requested, the concrete split is attached to each
       yielded source and scene.

    2. **Ingestion** — `ingest()` reads each source into one or more
       `(LazyFrame, MapContext)` pairs. The map context can be used to
       attach map information to the scene without including it in the raw
       data frame.

    3. **Pipeline** — `pipeline()` returns a composable processing pipeline
       that is applied to each `LazyFrame` produced by `ingest()`. The
       pipeline consists of a chain of transformations that process the raw
       data frame into the common schema.

    Notes
    -----
    - The map context can be: 1) a string (map key) that is later passed to
      the map resolver (if it exists), 2) an explicit resolver function that
      is attached to the scene, or 3) `None` which means no map information
      at this stage.

    - When no map resolver is attached in the processing steps, the
      `map_resolver()` method is used to provide a default resolver for the
      scene. This allows for flexibility in how map information is associated
      with scenes, and can accommodate datasets where map information is
      stored in different ways (e.g., separate files, embedded in the raw
      data, or not available at all). The default implementation is
      `no_map()`, which indicates that no map is available for the scenes
      produced by this loader.

    - There are priorities when it comes to the map key (if used). If it is
      directly attached to the source in the `all_sources()` step, that
      takes highest priority since it is the most explicit. If not, then if
      the map context returned by `ingest()` is a string, it is used as the
      map key. Finally, if neither of those are available, the map key will be
      `None` and the resolver (if any) will need to handle that case.

    """

    _shared_memory_name: ClassVar[dict[MapKey, str] | str | None] = None
    _split_methods: ClassVar[tuple[tuple[DatasetSplit, str], ...]] = (
        (DatasetSplit.TRAIN, "train_sources"),
        (DatasetSplit.VAL, "validate_sources"),
        (DatasetSplit.TEST, "test_sources"),
    )

    def __init__(
        self,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        *,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        enforce_schema: bool = True,
    ) -> None:
        """Initialize internal state.

        Parameters
        ----------
        loader_config : LoaderConfig, optional
            Configuration for the loader. If None, `default_config()` is used.
        enforce_schema : bool, optional
            Whether to enforce the scene schema on each created scene.
            Defaults to True.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split(s) to process. Can be a single `DatasetSplit`, an
            iterable of `DatasetSplit`, or `None` to indicate all splits.
            Defaults to `None`.

        """
        self._count: int = 0
        self._source_counter: int = 0
        self._enforce_schema: bool = enforce_schema
        self._loader_config: LoaderConfig = loader_config or self.default_config()
        self._map_config: MapConfig = map_config or self.default_map_config()
        self._pipeline: Pipeline | None = None
        if splits is None:
            self._splits: list[DatasetSplit] | None = None
        elif isinstance(splits, DatasetSplit):
            self._splits = [splits]
        else:
            self._splits = list(splits)

    def all_sources(self) -> Iterable[Source[SourceT]]:
        """Discover and yield sources across all available dataset partitions.

        For datasets with predefined splits this combines the implemented
        split-specific discovery hooks without assigning train/val/test
        metadata. Datasets without predefined splits should instead override
        `discover_sources()`.

        Yields
        ------
        Source[SourceT]
            Each raw data source to be processed.

        """
        supported_splits = self.supported_splits()
        if supported_splits:
            for split in supported_splits:
                yield from self._normalize_sources(self._raw_sources_for_split(split))
            return

        yield from self._normalize_sources(self.discover_sources())

    def discover_sources(self) -> Iterable[Source[SourceT]]:
        """Discover sources for datasets that do not define predefined splits."""
        msg = (
            f"{type(self).__name__} must implement discover_sources() or one or more "
            "split-specific source methods."
        )
        raise NotImplementedError(msg)

    @abstractmethod
    def ingest(self, source: Source[SourceT]) -> Iterable[IngestOutput]:
        """Read a raw data source into one or more `(LazyFrame, MapResolver)` pairs.

        Parameters
        ----------
        source : Source[SourceT]
            The raw data source to read.

        Yields
        ------
        tuple[pl.LazyFrame, MapResolver]
            A raw data frame paired with the resolver that should be
            attached to every scene derived from that frame.

        """

    @abstractmethod
    def pipeline(self) -> Pipeline:
        """Return the composable processing pipeline for this loader.

        The pipeline is a chain of transform steps that are applied to every
        `LazyFrame` produced by `ingest`.

        Returns
        -------
        Pipeline
            The processing pipeline.

        """

    @classmethod
    @abstractmethod
    def default_config(cls) -> LoaderConfig:
        """Return the default loader configuration for this dataset.

        This is a classmethod so that the default configuration can be inspected
        without constructing a loader instance.

        Returns
        -------
        LoaderConfig
            Default configuration for this loader.

        """

    @classmethod
    def default_map_config(cls) -> MapConfig:
        """Return the default map configuration for this dataset.

        Returns
        -------
        MapConfig
            Default map configuration for this loader.

        """
        return MapConfig.default()

    def train_sources(self) -> Iterable[Source[SourceT]]:
        """Return sources belonging to the **training** split.

        Override this method in subclasses whose underlying dataset provides
        a predefined training split.  The default implementation raises
        `SplitNotSupportedError`.

        Raises
        ------
        SplitNotSupportedError
            Always, unless overridden by a subclass that supports splits.

        """
        raise SplitNotSupportedError(type(self).__name__, DatasetSplit.TRAIN)

    def test_sources(self) -> Iterable[Source[SourceT]]:
        """Return sources belonging to the **test** split.

        Override this method in subclasses whose underlying dataset provides
        a predefined test split.  The default implementation raises
        `SplitNotSupportedError`.

        Raises
        ------
        SplitNotSupportedError
            Always, unless overridden by a subclass that supports splits.

        """
        raise SplitNotSupportedError(type(self).__name__, DatasetSplit.TEST)

    def validate_sources(self) -> Iterable[Source[SourceT]]:
        """Return sources belonging to the **validation** split.

        Override this method in subclasses whose underlying dataset provides
        a predefined validation split.  The default implementation raises
        `SplitNotSupportedError`.

        Raises
        ------
        SplitNotSupportedError
            Always, unless overridden by a subclass that supports splits.

        """
        raise SplitNotSupportedError(type(self).__name__, DatasetSplit.VAL)

    @override
    def sources(self) -> Iterable[Source[SourceT]]:
        """Return sources for the currently configured split.

        When no split selection is configured, all available sources are
        yielded without predefined split metadata. When explicit splits are
        requested, split-aware loaders assign the concrete train/val/test split
        to each source automatically.

        Yields
        ------
        Source[SourceT]
            Each raw data source to be processed for the configured split(s).

        """
        if self._splits is None:
            yield from self.all_sources()
            return

        for split in self._splits:
            yield from self._get_sources_for_split(split)

    @classmethod
    def supported_splits(cls) -> tuple[DatasetSplit, ...]:
        """Return the predefined splits explicitly implemented by this loader."""
        return tuple(
            split
            for split, method_name in cls._split_methods
            if getattr(cls, method_name) is not getattr(BaseSceneLoader, method_name)
        )

    @property
    def selected_splits(self) -> tuple[DatasetSplit, ...]:
        """Return the requested splits, or all supported predefined splits."""
        if self._splits is not None:
            return tuple(self._splits)
        return self.supported_splits()

    def _get_sources_for_split(self, split: DatasetSplit) -> Iterable[Source[SourceT]]:
        """Dispatch to the correct source method and assign the concrete split."""
        yield from self._normalize_sources(self._raw_sources_for_split(split), inferred_split=split)

    def _raw_sources_for_split(self, split: DatasetSplit) -> Iterable[Source[SourceT]]:
        """Dispatch to the correct raw source method based on a single split."""
        if split is DatasetSplit.TRAIN:
            return self.train_sources()
        if split is DatasetSplit.TEST:
            return self.test_sources()
        return self.validate_sources()

    def _normalize_sources(
        self,
        sources: Iterable[Source[SourceT]],
        *,
        inferred_split: DatasetSplit | None = None,
    ) -> Iterable[Source[SourceT]]:
        """Apply inferred split metadata to yielded sources."""
        for source in sources:
            yield self._resolve_source_split(source, inferred_split)

    @staticmethod
    def _resolve_source_split(
        source: Source[SourceT],
        inferred_split: DatasetSplit | None,
    ) -> Source[SourceT]:
        """Resolve the effective split assignment for a discovered source."""
        if inferred_split is None:
            resolved_split = source.split_assignment
        elif source.split_assignment is None:
            resolved_split = inferred_split
        elif source.split_assignment_override:
            resolved_split = source.split_assignment
        else:
            resolved_split = inferred_split

        if source.split_assignment == resolved_split:
            return source
        return source.with_split_assignment(resolved_split)

    def map_resolver(self) -> MapResolver:  # noqa: PLR6301
        """Return a resolver for this dataset's map data.

        If no resolver is attached during ingestion, this dataset-level resolver
        is attached to created scenes and used to resolve their `map_key`.

        Returns
        -------
        MapResolver
            A callable that resolves map keys. The default returns `no_map`.

        """
        return no_map()

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
        self._count = 0
        self._source_counter = 0
        for source in self.sources():
            self._source_counter += 1
            for scene_df, resolver in self.process_next(source):
                yield self.create_scene(scene_df, source, resolver)

    @override
    def create_scene(
        self,
        df: pl.DataFrame,
        source: Source[SourceT],
        resolver: MapContext | None = None,
        scene_number: int | None = None,
        split: DatasetSplit | None = None,
    ) -> Scene:
        """Create a Scene object from the processed DataFrame and its source.

        This method also calls `Scene.enforce_schema()` if
        `self._enforce_schema` is True to ensure the scene follows the expected
        schema. If overriding this method, make sure to follow the expected
        behavior regarding schema enforcement.

        Parameters
        ----------
        df : pl.DataFrame
            Processed DataFrame for the scene, expected to follow the
            common schema.
        source : Source[SourceT]
            The originating source.  The scene inherits
            `source.identifier` and `source.map_key`.
        resolver : MapResolver, optional
            The resolver to attach to the scene.  When `None`, falls
            back to `self.map_resolver()`.

        Returns
        -------
        Scene
            The created scene object.

        """
        map_key = source.map_key or (resolver if isinstance(resolver, str) else None)
        resolver = resolver if not isinstance(resolver, str) else self.map_resolver()
        scene = Scene(
            inner=df,
            scene_number=scene_number if scene_number is not None else self._count,
            input_len=self.input_len,
            output_len=self.output_len,
            map_key=map_key,
            map_resolver=resolver,
            split_assignment=split if split is not None else source.split_assignment,
        )
        self._count += 1
        return scene if not self._enforce_schema else scene.enforce_schema()

    @override
    def num_scenes(self) -> int | None:
        return None

    @override
    def num_sources(self) -> int | None:
        return None

    @staticmethod
    def derivative_names() -> dict[int, list[str]]:
        """Return the names of the derivatives for velocity and acceleration.

        Returns
        -------
        dict[int, list[str]]
            Mapping of derivative order to column names.

        """
        return {
            1: ["vx", "vy"],
            2: ["ax", "ay"],
        }

    @property
    def loader_config(self) -> LoaderConfig:
        """Return the loader configuration."""
        return self._loader_config

    @property
    def map_config(self) -> MapConfig:
        """Return the map configuration."""
        return self._map_config

    @property
    def original_input_len(self) -> int:
        """Original observation length in frames (before resampling)."""
        return self.loader_config.input_len

    @property
    def original_output_len(self) -> int:
        """Original prediction length in frames (before resampling)."""
        return self.loader_config.output_len

    @property
    def sequence_length(self) -> int:
        """Total sequence length (observation + prediction) in frames."""
        return self.loader_config.input_len + self.loader_config.output_len

    @property
    def input_len(self) -> int:
        """Observation length in frames (resulting value in Scene)."""
        up, down = (
            self.loader_config.resampling.factors if self.loader_config.resampling else (1, 1)
        )
        ratio = up / down
        return int((self.original_input_len - 1) * ratio + 1)

    @property
    def output_len(self) -> int:
        """Prediction length in frames (resulting value in Scene)."""
        up, down = (
            self.loader_config.resampling.factors if self.loader_config.resampling else (1, 1)
        )
        ratio = up / down
        total_len = int((self.sequence_length - 1) * ratio + 1)
        return total_len - self.input_len

    @property
    def post_sample_time(self) -> float:
        """Time interval between frames after resampling."""
        if self.loader_config.resampling is None:
            return self.loader_config.sample_time
        up, down = self.loader_config.resampling.factors
        ratio = up / down
        return self.loader_config.sample_time / ratio

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

    @classmethod
    def set_shared_memory(
        cls,
        name: str | None = None,
        mappings: dict[MapKey, str] | None = None,
    ) -> None:
        """Set the shared memory name or mapping for this loader class.

        This is used to share data (e.g., map graphs) between processes when
        using multiprocessing. The `name` parameter sets a single shared memory
        name for all map keys, while the `mappings` parameter allows for
        specifying different shared memory names for different map keys.

        Parameters
        ----------
        name : str, optional
            A single shared memory name to use for all map keys. If provided,
            this will override any existing mappings.
        mappings : dict[MapKey, str], optional
            A dictionary mapping specific map keys to shared memory names. If
            provided, this will override any existing mappings and the `name`
            parameter. When both *name* and *mappings* are omitted, any
            existing shared-memory configuration is cleared.

        """
        if mappings is not None and name is not None:
            msg = "Cannot set both 'name' and 'mappings'. Please choose one."
            raise ValueError(msg)
        if mappings is not None:
            cls._shared_memory_name = mappings
        elif name is not None:
            cls._shared_memory_name = name
        else:
            cls._shared_memory_name = None

    @staticmethod
    def _normalize_data_root(data_root: Path | str) -> Path:
        """Normalize a filesystem input into a `Path` instance."""
        return Path(data_root)

    @staticmethod
    def _count_matching_files(
        directories: Iterable[Path],
        pattern: str,
        *,
        recursive: bool = False,
    ) -> int:
        """Count files matching a glob pattern across existing directories only."""
        total = 0
        for directory in directories:
            if not directory.is_dir():
                continue
            matches = directory.rglob(pattern) if recursive else directory.glob(pattern)
            total += sum(1 for _ in matches)
        return total

    def _invalid_loader_argument(self, detail: str) -> ValueError:
        """Create a consistent loader configuration error."""
        return LoaderConfigError(f"{type(self).__name__}: {detail}")
