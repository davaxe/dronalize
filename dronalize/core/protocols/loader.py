from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (  # pyright: ignore[reportUnusedImport]
    Callable,
    Hashable,
    Iterable,
)
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
)

from typing_extensions import override

from dronalize.core.datatypes.scene import Scene
from dronalize.core.pipeline import Pipeline

if TYPE_CHECKING:
    import polars as pl

    from dronalize.core.datatypes import LoaderConfig
    from dronalize.core.datatypes.map_context import MapContext


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
    map_context: MapContext | None = None
    """Optional map context associated with the source.

    This can be useful if all scenes generated from this source share the same
    map context.
    """
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
    Identifier type is used to unique identify each scene source (e.g., str,
    int).

    Processing model
    ----------------
    Each source is processed through these stages:

    1. **sources()** — discover raw data items.
    2. **load_raw()** — ingest each source into `(LazyFrame, MapContext)`
       pairs.
    3. **pipeline()** — a composable :class:`~dronalize.core.pipeline.Pipeline`
       of `LazyFrame → LazyFrame` transforms (filtering, resampling,
       derivatives, windowing, …).
    4. **create_scene()** — wrap the final DataFrame into a :class:`Scene`.

    Subclasses **must** implement :meth:`sources`, :meth:`load_raw`, and
    :meth:`default_config`.

    For the processing step, subclasses can *either*:

    * Override :meth:`pipeline` to return a composable
      :class:`~dronalize.core.pipeline.Pipeline` (preferred), **or**
    * Override :meth:`normalize` for simple 1:1 transforms (legacy).

    If :meth:`pipeline` returns a non-empty pipeline, it is used and
    :meth:`normalize` is ignored.  If :meth:`pipeline` returns an empty
    pipeline, :meth:`normalize` is called as a fallback for backward
    compatibility.
    """

    def __init__(
        self,
        loader_config: LoaderConfig | None = None,
        *,
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

        """
        self._count: int = 0
        self._source_counter: int = 0
        self._enforce_schema: bool = enforce_schema
        self._loader_config: LoaderConfig | None = loader_config

    # --- Abstract Steps (The "Blanks" to fill) ---

    @abstractmethod
    def sources(self) -> Iterable[Source[IdT, SourceT]]:
        """Discover and yield identifiers for each scene to process.

        Yields
        ------
        Source[IdT, SourceT]
            Each raw data source to be processed.

        """

    @abstractmethod
    def load_raw(self, source: Source[IdT, SourceT]) -> Iterable[tuple[pl.LazyFrame, MapContext]]:
        """Read the raw data source into one or more DataFrame(s) (any schema).

        Parameters
        ----------
        source : Source[IdT, SourceT]
            The raw data source to load.

        Yields
        ------
        tuple[pl.LazyFrame, MapContext]
            Raw data frame and its associated map context.

        """

    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Convert the raw DataFrame into the common schema.

        .. deprecated::
            Override :meth:`pipeline` instead.  This method is only called
            as a fallback when :meth:`pipeline` returns an empty pipeline.

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

    def pipeline(self) -> Pipeline:
        """Return the composable processing pipeline for this loader.

        The pipeline is a chain of :class:`~dronalize.core.pipeline.Transform`
        (1:1) and :class:`~dronalize.core.pipeline.FanOut` (1:N) steps that
        are applied to every LazyFrame produced by :meth:`load_raw`.

        The default implementation returns an **empty** pipeline, which
        causes :meth:`process_next` to fall back to :meth:`normalize`.
        Override this method to define a composable pipeline.

        Returns
        -------
        Pipeline
            The processing pipeline.  An empty pipeline triggers the
            legacy :meth:`normalize` fallback.

        Example
        -------
        ::

            from dronalize.core import transforms as T
            from dronalize.core.pipeline import Pipeline

            def pipeline(self) -> Pipeline:
                return (
                    Pipeline()
                    .then(T.filter(self.loader_config))
                    .then_flat_map(T.window(self.loader_config))
                    .then(T.resample(
                        self.loader_config,
                        add_derivative=True,
                        add_second_derivative=True,
                        derivative_rename=self.derivative_names(),
                    ))
                    .then(T.yaw_from_vel())
                )

        """
        return Pipeline()

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

    @override
    def scenes_callback(
        self,
        callback: Callable[Concatenate[Scene[IdT], P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        for scene in self.scenes():
            callback(scene, *args, **kwargs)

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

        This is different from `num_scenes()` since each source can potentially
        generate multiple scenes (e.g., by using sliding window sampling). In
        some cases this can be expensive to compute or not known in advance, in
        that case `None` is returned.

        This could trivially be implemented as `len(list(self.sources()))`, but
        that would require loading all sources into memory which can be
        expensive for large datasets.

        Returns
        -------
        int or None
            Total number of sources, or `None` if not known in advance.

        """
        _self = self  # To satisfy Ruff, since `self` will most likely be used in subclasses.
        return None

    def process_next(
        self,
        source: Source[IdT, SourceT],
    ) -> Iterable[tuple[pl.DataFrame, MapContext]]:
        """Process a single data item through the pipeline.

        If :meth:`pipeline` returns a non-empty pipeline, each raw
        LazyFrame is run through that pipeline (which may fan out into
        multiple frames).  Otherwise, :meth:`normalize` is called as a
        1:1 fallback.

        Parameters
        ----------
        source : Source[IdT, SourceT]
            The source to process.

        Yields
        ------
        tuple[pl.DataFrame, MapContext]
            Processed data frame and its associated map context.

        """
        pipe = self.pipeline()
        use_pipeline = bool(pipe)

        def _collect(lf: pl.LazyFrame) -> pl.DataFrame | None:
            try:
                df = lf.collect()
            except ValueError:
                return None
            return df if not df.is_empty() else None

        for raw_df, map_context in self.load_raw(source):
            if use_pipeline:
                for result_lf in pipe.execute(raw_df):
                    df = _collect(result_lf)
                    if df is not None:
                        yield df, map_context
            else:
                # Legacy path: single 1:1 normalize call
                df = _collect(self.normalize(raw_df))
                if df is not None:
                    yield df, map_context

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
            for scene_df, map_context in self.process_next(source):
                yield self.create_scene(scene_df, source.identifier, map_context)

    def create_scene(self, df: pl.DataFrame, source_id: IdT, map_context: MapContext) -> Scene[IdT]:
        """Create a Scene object from the processed DataFrame and source identifier.

        This method also calls `Scene.enforce_schema()` if
        `self._enforce_schema` is True to ensure the scene follows the expected
        schema. If overriding this method, make sure to follow the expected
        behavior regarding schema enforcement.

        Parameters
        ----------
        df : pl.DataFrame
            Processed DataFrame for the scene, expected to follow the common schema.
        source_id : IdT
            Identifier for the scene source (e.g., file name, index).
        map_context : MapContext
            Map context associated with the scene.

        Returns
        -------
        Scene[IdT]
            The created scene object.

        """
        scene = Scene(
            inner=df,
            identifier=source_id,
            scene_number=self._count,
            input_len=self.input_len,
            output_len=self.output_len,
            map_context=map_context,
        )
        self._count += 1
        return scene if not self._enforce_schema else scene.enforce_schema()

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
