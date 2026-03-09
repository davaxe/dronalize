from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Concatenate, Generic, Protocol

import polars as pl
from typing_extensions import override

from dronalize.core._types import IdT, P, SourceT
from dronalize.core.datatypes.map_resolver import MapKey, MapResolver, no_map
from dronalize.core.datatypes.scene import Scene
from dronalize.core.datatypes.split import DatasetSplit, SplitNotSupportedError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dronalize.core.datatypes import LoaderConfig
    from dronalize.core.protocols.writer import SceneWriter
    from dronalize.pipeline.pipeline import Pipeline


MapContext = MapResolver | MapKey | None
IngestOutput = tuple[pl.LazyFrame, MapContext]


@dataclass(slots=True, frozen=True)
class Source(Generic[IdT, SourceT]):
    """Represents a raw data source for a scene, identified by a unique identifier."""

    identifier: IdT
    """Generic identifier for the source, e.g., file name, URL, database key."""
    inner: SourceT
    """The actual source data, which can be of any type (e.g., file path, raw data)."""
    map_key: MapKey = None
    """Optional map key associated with this source."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata associated with the source."""


class SceneLoader(Protocol, Generic[IdT]):
    """Minimal protocol for scene loading, used for type hinting."""

    def scenes(self) -> Iterable[Scene[IdT]]:
        """Process scenes and yield them one by one.

        This is the main method for processing scenes. It yields `Scene` objects
        one at a time, allowing for memory-efficient processing of large datasets.

        Yields
        ------
        Scene[IdT]
            Each processed scene, with its identifier and associated data.

        """
        ...

    def scenes_callback(
        self,
        callback: Callable[Concatenate[Scene[IdT], P], None],
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

    def write_scenes(
        self,
        writer_factory: Callable[..., SceneWriter],
        finalize: Callable[[SceneWriter], None],
    ) -> None:
        """Process scenes and write them using the provided writer factory.

        This method provides a convenient way to process scenes and write them
        using a `SceneWriter`. The `writer_factory` is called once to create a
        `SceneWriter` instance, which is then used to write each processed scene.
        After all scenes have been processed, the `finalize` function is called
        with the writer instance to perform any necessary cleanup or finalization
        steps.

        Parameters
        ----------
        writer_factory : Callable
            A factory function that takes additional arguments and returns a
            `SceneWriter` instance for writing scenes.
        finalize : Callable
            A function that takes the `SceneWriter` instance and performs any
            necessary finalization steps after all scenes have been written.

        """


class ProcessableLoader(Protocol, Generic[IdT, SourceT]):
    """Minimal protocol required to work with a processor abstraction.

    This protocol defines the essential interface for discovering data sources,
    tracking dataset sizes, processing raw sources into tabular data, and
    constructing final scene objects.

    """

    def sources(self) -> Iterable[Source[IdT, SourceT]]:
        """Discover and yield the data sources to be processed.

        Returns
        -------
        Iterable[Source[IdT, SourceT]]
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

    def process_next(
        self, source: Source[IdT, SourceT]
    ) -> Iterable[tuple[pl.DataFrame, MapContext]]:
        """Process a single raw data source into data frames and map contexts.

        Parameters
        ----------
        source : Source[IdT, SourceT]
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
        source: Source[IdT, SourceT],
        resolver: MapContext | None = None,
        scene_number: int | None = None,
    ) -> Scene[IdT]:
        """Construct a Scene object from processed data.

        Parameters
        ----------
        df : pl.DataFrame
            The processed data frame containing the scene data.
        source : Source[IdT, SourceT]
            The originating raw data source.
        resolver : MapContext | None, optional
            The map context or resolver associated with the scene.
        scene_number : int | None, optional
            An optional numeric index or identifier for the generated scene.

        Returns
        -------
        Scene[IdT]
            The fully constructed scene object.
        """
        ...


class BaseSceneLoader(ABC, SceneLoader[IdT], ProcessableLoader[IdT, SourceT]):
    """ABC interface for processing raw data sources into a standardized format.

    This class contains logic for orchestrating the loading and processing of
    raw data sources into scenes, while allowing subclasses to implement the
    specific details of how raw data is read and processed. The main processing
    flow is as follows:

    1. **Source discovery** — `sources()` dispatches to the appropriate source
       method based on the `split` parameter:

       - `DatasetSplit.ALL` (or `None`) → `all_sources()` *(abstract —
         every subclass must implement this)*
       - `DatasetSplit.TRAIN` → `train_sources()`
       - `DatasetSplit.VALIDATE` → `validate_sources()`
       - `DatasetSplit.TEST` → `test_sources()`

       The `train_sources`, `test_sources`, and `validate_sources`
       methods are **optionally overridable**.  Their default implementations
       raise `SplitNotSupportedError`, so datasets that do not ship with
       predefined splits need only implement `all_sources()`.  Datasets
       that *do* have splits override the relevant methods and typically
       implement `all_sources()` by chaining all three split methods.

    2. **Ingestion** — `ingest()` reads each source into one or more
       `(LazyFrame, MapContext)` pairs. The map context can be used to
       attach map information to the scene without including it in the raw
       data frame.

    3. **Pipeline** — `pipeline()` returns a composable processing pipeline
       that is applied to each `LazyFrame` produced by `ingest()`. The
       pipeline consists of a chain of transformations that process the raw
       data frame into the common schema.

    Considerations
    --------------
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

    _shared_memory_name: ClassVar[str | None] = None

    def __init__(
        self,
        loader_config: LoaderConfig | None = None,
        *,
        enforce_schema: bool = True,
        split: DatasetSplit | None = None,
    ) -> None:
        """Initialize internal state.

        Parameters
        ----------
        loader_config : LoaderConfig, optional
            Configuration for the loader. If None, `default_config()` is used.
        enforce_schema : bool, optional
            Whether to enforce the scene schema on each created scene.
            Defaults to True.
        split : DatasetSplit, optional
            The dataset split to load.  When `None` or
            `DatasetSplit.ALL`, all data is loaded via `all_sources()`.
            When set to `TRAIN`, `TEST`, or `VALIDATE`, the
            corresponding `train_sources()`, `test_sources()`, or
            `validate_sources()` method is called.  Loaders that do not
            support splits will raise `SplitNotSupportedError` for any
            value other than `None` / `ALL`.

        """
        self._count: int = 0
        self._source_counter: int = 0
        self._enforce_schema: bool = enforce_schema
        self._loader_config: LoaderConfig = loader_config or self.default_config()
        self._pipeline: Pipeline | None = None
        self._split: DatasetSplit = split if split is not None else DatasetSplit.ALL

    # ===================================================================
    # Abstract — every subclass must implement these
    # ===================================================================

    @abstractmethod
    def all_sources(self) -> Iterable[Source[IdT, SourceT]]:
        """Discover and yield identifiers for **all** scenes to process.

        Every subclass must implement this method.  It is called when no
        specific split is requested (i.e. `split` is `None` or
        `DatasetSplit.ALL`).

        Ideally, this should be lightweight and not do any heavy processing.

        Yields
        ------
        Source[IdT, SourceT]
            Each raw data source to be processed.

        """

    @abstractmethod
    def ingest(self, source: Source[IdT, SourceT]) -> Iterable[IngestOutput]:
        """Read a raw data source into one or more `(LazyFrame, MapResolver)` pairs.

        Parameters
        ----------
        source : Source[IdT, SourceT]
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
        """Return the default processor configuration for this dataset.

        This is a classmethod so that the default configuration can be inspected
        without constructing a loader instance.

        Returns
        -------
        LoaderConfig
            Default configuration for this loader.

        """

    def train_sources(self) -> Iterable[Source[IdT, SourceT]]:
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

    def test_sources(self) -> Iterable[Source[IdT, SourceT]]:
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

    def validate_sources(self) -> Iterable[Source[IdT, SourceT]]:
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
    def sources(self) -> Iterable[Source[IdT, SourceT]]:
        """Return sources for the currently configured split.

        This method dispatches to `all_sources()`,
        `train_sources()`, `test_sources()`, or
        `validate_sources()` based on the `split` value passed at
        construction time.

        Returns
        -------
        Iterable[Source[IdT, SourceT]]
            The sources for the active split.

        """
        if self._split is DatasetSplit.ALL:
            return self.all_sources()
        if self._split is DatasetSplit.TRAIN:
            return self.train_sources()
        if self._split is DatasetSplit.TEST:
            return self.test_sources()
        if self._split is DatasetSplit.VAL:
            return self.validate_sources()
        return self.all_sources()

    def map_resolver(self) -> MapResolver:
        """Return a resolver for this dataset's map data.

        If the `resolver` is not set anywhere else, this resolver will be
        attached to the scene and can be used to resolve

        Returns
        -------
        MapResolver
            A callable that resolves map keys. The default returns `no_map`.

        """
        _self = self
        return no_map()

    def set_loader_config(self, config: LoaderConfig) -> None:
        """Set the loader configuration.

        This can be used to update the loader's configuration after it has been
        constructed.  Note that changing the configuration may not have any
        effect if scenes have already been processed, since some configuration
        values (e.g., input/output lengths) are determined at initialization
        time.

        Parameters
        ----------
        config : LoaderConfig
            The new configuration to set.

        """
        self._loader_config = config

    @override
    def scenes_callback(
        self,
        callback: Callable[Concatenate[Scene[IdT], P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        for scene in self.scenes():
            callback(scene, *args, **kwargs)

    @override
    def scenes(self) -> Iterable[Scene[IdT]]:
        self._count = 0
        self._source_counter = 0
        for source in self.sources():
            self._source_counter += 1
            for scene_df, resolver in self.process_next(source):
                yield self.create_scene(scene_df, source, resolver)

    @override
    def write_scenes(
        self,
        writer_factory: Callable[P, SceneWriter],
        finalize: Callable[[SceneWriter], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        writer = writer_factory(*args, **kwargs)
        try:
            for scene in self.scenes():
                writer.write(scene)
        finally:
            finalize(writer)

    @override
    def create_scene(
        self,
        df: pl.DataFrame,
        source: Source[IdT, SourceT],
        resolver: MapContext | None = None,
        scene_number: int | None = None,
    ) -> Scene[IdT]:
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
        source : Source[IdT, SourceT]
            The originating source.  The scene inherits
            `source.identifier` and `source.map_key`.
        resolver : MapResolver, optional
            The resolver to attach to the scene.  When `None`, falls
            back to `self.map_resolver()`.

        Returns
        -------
        Scene[IdT]
            The created scene object.

        """
        map_key = source.map_key or (resolver if isinstance(resolver, str) else None)
        resolver = resolver if not isinstance(resolver, str) else self.map_resolver()
        scene = Scene(
            inner=df,
            identifier=source.identifier,
            scene_number=scene_number if scene_number is not None else self._count,
            input_len=self.input_len,
            output_len=self.output_len,
            map_key=map_key,
            map_resolver=resolver,
        )
        self._count += 1
        return scene if not self._enforce_schema else scene.enforce_schema()

    # ===================================================================
    # Convenience / introspection
    # ===================================================================

    @override
    def num_scenes(self) -> int | None:
        _self = self
        return None

    @override
    def num_sources(self) -> int | None:
        _self = self
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
        return self._loader_config or self.default_config()

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
        source: Source[IdT, SourceT],
    ) -> Iterable[tuple[pl.DataFrame, MapContext]]:
        if self._pipeline is None:
            self._pipeline = self.pipeline()

        for raw_lf, map_context in self.ingest(source):
            for df in self._pipeline.execute(raw_lf, collect=True, filter_empty=True):
                yield df, map_context

    @classmethod
    def set_shared_memory(cls, name: str) -> None:
        """Set the name of the shared memory segment to use for this loader.

        This memory could be used for anything, but is mainly meant for sharing
        Maps across processes without needing to serialize them with each scene.

        Parameters
        ----------
        name : str
            The name of the shared memory segment to use.
        """
        cls._shared_memory_name = name
