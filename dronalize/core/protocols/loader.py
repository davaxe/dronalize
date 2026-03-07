from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Concatenate, Generic, ParamSpec, Protocol, TypeAlias, TypeVar

import polars as pl
from typing_extensions import override

from dronalize.core.datatypes.map_context import MapKey, MapResolver, no_map
from dronalize.core.datatypes.scene import Scene
from dronalize.core.datatypes.split import DatasetSplit, SplitNotSupportedError

if TYPE_CHECKING:
    from dronalize.core.datatypes import LoaderConfig
    from dronalize.core.pipeline import Pipeline


MapContext: TypeAlias = MapResolver | MapKey | None
IngestOutput: TypeAlias = tuple[pl.LazyFrame, MapContext]

SourceT = TypeVar("SourceT")
SourceT_co = TypeVar("SourceT_co", covariant=True)
IdT = TypeVar("IdT", bound=Hashable)
P = ParamSpec("P")


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
        """Yield processed scenes one at a time."""
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


class BaseSceneLoader(ABC, SceneLoader[IdT], Generic[IdT, SourceT]):
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
        self._loader_config: LoaderConfig | None = loader_config
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
        raise SplitNotSupportedError(type(self).__name__, DatasetSplit.VALIDATE)

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
        if self._split is DatasetSplit.VALIDATE:
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

    @override
    def scenes_callback(
        self,
        callback: Callable[Concatenate[Scene[IdT], P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        for scene in self.scenes():
            callback(scene, *args, **kwargs)

    def scenes(self) -> Iterable[Scene[IdT]]:
        """Iterate over all sources and process them into scenes.

        This will lazy-load and process each source one at a time. Counters are
        reset at the start of each call so a loader instance can be iterated
        more than once.

        Yields
        ------
        Scene[IdT]
            Processed scenes one at a time.

        """
        self._count = 0
        self._source_counter = 0
        for source in self.sources():
            self._source_counter += 1
            for scene_df, resolver in self.process_next(source):
                yield self.create_scene(scene_df, source, resolver)

    def create_scene(
        self,
        df: pl.DataFrame,
        source: Source[IdT, SourceT],
        resolver: MapContext | None = None,
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
            scene_number=self._count,
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

    def num_scenes(self) -> int | None:
        """Get the total number of scenes that will be processed.

        In some cases this can be expensive to compute or not known in advance,
        in that case `None` is returned.

        Returns
        -------
        int or None
            Total number of scenes, or `None` if not known in advance.

        """
        _self = self
        return None

    def num_sources(self) -> int | None:
        """Get the total number of sources that will be processed.

        This is different from `num_scenes()` since each source can potentially
        generate multiple scenes (e.g., by using sliding window sampling). In
        some cases this can be expensive to compute or not known in advance, in
        that case `None` is returned.

        Returns
        -------
        int or None
            Total number of sources, or `None` if not known in advance.

        """
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

    def process_next(
        self,
        source: Source[IdT, SourceT],
    ) -> Iterable[tuple[pl.DataFrame, MapContext]]:
        """Process next source.

        Parameters
        ----------
        source : Source[IdT, SourceT]
            The source to process.

        Yields
        ------
        tuple[pl.DataFrame, MapResolver]
            Processed data frames, each paired with its resolver.

        """
        if self._pipeline is None:
            self._pipeline = self.pipeline()

        for raw_lf, map_context in self.ingest(source):
            for df in self._pipeline.execute(raw_lf, collect=True, filter_empty=True):
                yield df, map_context
