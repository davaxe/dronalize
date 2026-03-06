from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Concatenate, Generic, ParamSpec, Protocol, TypeVar

from typing_extensions import override

from dronalize.core.datatypes.map_context import MapKey, MapResolver, no_map
from dronalize.core.datatypes.scene import Scene
from dronalize.core.pipeline import Pipeline

if TYPE_CHECKING:
    import polars as pl

    from dronalize.core.datatypes import LoaderConfig


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

    Generic over the data scene source type and identifier type. Source type can
    be anything (e.g., file path, URL, database connection, raw data).
    Identifier type is used to uniquely identify each scene source (e.g., str,
    int).
    """

    def __init__(
        self,
        loader_config: LoaderConfig | None = None,
        *,
        enforce_schema: bool = True,
        use_pipeline: bool = True,
    ) -> None:
        """Initialize internal state.

        Parameters
        ----------
        loader_config : LoaderConfig, optional
            Configuration for the loader. If None, `default_config()` is used.
        enforce_schema : bool, optional
            Whether to enforce the scene schema on each created scene.
            Defaults to True.

        """
        self._count: int = 0
        self._source_counter: int = 0
        self._enforce_schema: bool = enforce_schema
        self._loader_config: LoaderConfig | None = loader_config
        self._use_pipeline: bool | None = use_pipeline

    # ===================================================================
    # Abstract — every subclass must implement these
    # ===================================================================

    @abstractmethod
    def sources(self) -> Iterable[Source[IdT, SourceT]]:
        """Discover and yield identifiers for each scene to process.

        Yields
        ------
        Source[IdT, SourceT]
            Each raw data source to be processed.

        """

    @classmethod
    @abstractmethod
    def default_config(cls) -> LoaderConfig:
        """Return the default processor configuration for this dataset.

        This is a classmethod so that the default configuration can be
        inspected without constructing a loader instance.

        Returns
        -------
        LoaderConfig
            Default configuration for this loader.

        """

    # ===================================================================
    # New path — ingest + pipeline  (override these for new loaders)
    # ===================================================================

    def ingest(self, source: Source[IdT, SourceT]) -> Iterable[pl.LazyFrame]:
        """Read a raw data source into one or more `LazyFrame` (s).

        This is the **new-path** equivalent of `load_raw`.  It is
        responsible *only* for reading / parsing the data — all
        filtering, resampling, windowing, etc. belong in `pipeline`.

        The default implementation delegates to `load_raw` (stripping
        the legacy `MapKey` second element) so that existing loaders
        continue to work without changes when they have not yet been
        migrated.

        Parameters
        ----------
        source : Source[IdT, SourceT]
            The raw data source to read.

        Yields
        ------
        pl.LazyFrame
            One or more raw data frames.

        """
        for lf, _map_ctx in self.load_raw(source):
            yield lf

    def pipeline(self) -> Pipeline:
        """Return the composable processing pipeline for this loader.

        The pipeline is a chain of `~dronalize.core.pipeline.Transform`
        (1:1) and `~dronalize.core.pipeline.FlatMapTransform` (1:N)
        steps that are applied to every `LazyFrame` produced by
        `ingest`.

        The default implementation returns an **empty** pipeline, which
        causes `process_next` to fall back to the legacy
        `load_raw` + `normalize` path.

        Returns
        -------
        Pipeline
            The processing pipeline.  An empty pipeline triggers the
            legacy fallback.

        """
        return Pipeline()

    def map_resolver(self) -> MapResolver:
        """Return a resolver for this dataset's map data.

        Returns
        -------
        MapResolver
            A callable that resolves map keys. The default returns `no_map`.

        """
        return no_map()

    # ===================================================================
    # Legacy path — load_raw + normalize  (existing loaders use these)
    # ===================================================================

    def load_raw(
        self,
        source: Source[IdT, SourceT],
    ) -> Iterable[tuple[pl.LazyFrame, MapKey]]:
        """Read the raw data source into one or more DataFrame(s) (any schema).

        .. deprecated::
            Override `ingest` and `pipeline` instead.
            This method is only called as a fallback when `pipeline()`
            returns an empty pipeline and `ingest` has not been
            overridden.

        Parameters
        ----------
        source : Source[IdT, SourceT]
            The raw data source to load.

        Yields
        ------
        tuple[pl.LazyFrame, MapKey]
            Raw data frame and an associated map key.

        """
        msg = (
            f"{type(self).__name__} must implement either `ingest` (new path) "
            f"or `load_raw` (legacy path)."
        )
        raise NotImplementedError(msg)

    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Convert the raw DataFrame into the common schema.

        .. deprecated::
            Override `pipeline` instead.  This method is only called
            as a fallback when `pipeline` returns an empty pipeline.

        The default implementation is the identity (returns *df* unchanged).

        Parameters
        ----------
        df : pl.LazyFrame
            Raw data frame to normalize.

        Returns
        -------
        pl.LazyFrame
            Normalized data frame following the common schema.

        """
        return df

    # ===================================================================
    # Orchestration
    # ===================================================================

    def process_next(
        self,
        source: Source[IdT, SourceT],
    ) -> Iterable[pl.DataFrame]:
        """Process a single data source through the appropriate path.

        * **New path:** `ingest` → `pipeline`  (when `pipeline`
          is non-empty).
        * **Legacy path:** `load_raw` → `normalize`  (when
          `pipeline` is empty).

        Parameters
        ----------
        source : Source[IdT, SourceT]
            The source to process.

        Yields
        ------
        pl.DataFrame
            Processed data frames, one per resulting scene.

        """
        pipe = self.pipeline()
        use_new_path = self._use_pipeline if self._use_pipeline is not None else bool(pipe)

        if use_new_path:
            yield from self._process_new_path(source, pipe)
        else:
            yield from self._process_legacy_path(source)

    def _process_new_path(
        self,
        source: Source[IdT, SourceT],
        pipe: Pipeline,
    ) -> Iterable[pl.DataFrame]:
        """Process next (new path).

        Parameters
        ----------
        source : Source[IdT, SourceT]
            The source to process.
        pipe : Pipeline
            A non-empty pipeline to apply.

        Yields
        ------
        pl.DataFrame
            Processed data frames.

        """
        for raw_lf in self.ingest(source):
            yield from pipe.execute(raw_lf, collect=True, filter_empty=True)

    def _process_legacy_path(
        self,
        source: Source[IdT, SourceT],
    ) -> Iterable[pl.DataFrame]:
        """Legacy path: load_raw → normalize.

        Parameters
        ----------
        source : Source[IdT, SourceT]
            The source to process.

        Yields
        ------
        pl.DataFrame
            Processed data frames.

        """
        for raw_lf, _map_key in self.load_raw(source):
            df = self._try_collect(self.normalize(raw_lf))
            if df is not None:
                yield df

    @staticmethod
    def _try_collect(lf: pl.LazyFrame) -> pl.DataFrame | None:
        """Collect a `LazyFrame`, returning `None` on empty or error."""
        try:
            df = lf.collect()
        except ValueError:
            return None
        return df if not df.is_empty() else None

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

        This will lazy-load and process each source one at a time.
        Counters are reset at the start of each call so a loader instance
        can be iterated more than once.

        Yields
        ------
        Scene[IdT]
            Processed scenes one at a time.

        """
        self._count = 0
        self._source_counter = 0
        for source in self.sources():
            self._source_counter += 1
            for scene_df in self.process_next(source):
                yield self.create_scene(scene_df, source)

    def create_scene(
        self,
        df: pl.DataFrame,
        source: Source[IdT, SourceT],
    ) -> Scene[IdT]:
        """Create a Scene object from the processed DataFrame and its source.

        This method also calls `Scene.enforce_schema()` if
        `self._enforce_schema` is True to ensure the scene follows the
        expected schema. If overriding this method, make sure to follow
        the expected behavior regarding schema enforcement.

        Parameters
        ----------
        df : pl.DataFrame
            Processed DataFrame for the scene, expected to follow the
            common schema.
        source : Source[IdT, SourceT]
            The originating source.  The scene inherits
            `source.identifier` and `source.map_key`.

        Returns
        -------
        Scene[IdT]
            The created scene object.

        """
        scene = Scene(
            inner=df,
            identifier=source.identifier,
            scene_number=self._count,
            input_len=self.input_len,
            output_len=self.output_len,
            map_key=source.map_key,
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
        _self = self  # To satisfy Ruff, since `self` will most likely be used in subclasses.
        return None

    def num_sources(self) -> int | None:
        """Get the total number of sources that will be processed.

        This is different from `num_scenes()` since each source can
        potentially generate multiple scenes (e.g., by using sliding
        window sampling). In some cases this can be expensive to compute
        or not known in advance, in that case `None` is returned.

        Returns
        -------
        int or None
            Total number of sources, or `None` if not known in advance.

        """
        _self = self  # To satisfy Ruff, since `self` will most likely be used in subclasses.
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
