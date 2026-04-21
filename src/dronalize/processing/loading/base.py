"""Base loader implementation for thin dataset-ingestion adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, cast

from typing_extensions import Self, TypeVar

from dronalize.core.errors import SplitNotSupportedError
from dronalize.core.typing import SourceT
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.loading.options import DatasetOptionsModel, NoDatasetOptions
from dronalize.processing.loading.resources import DatasetResources
from dronalize.processing.maps.resolver import MapResolver, no_map
from dronalize.processing.models import PipelinePlan, SplitRequest
from dronalize.processing.pipeline import spec
from dronalize.processing.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.config.models import MapConfig, ScenesConfig, ScreeningConfig
    from dronalize.core.categories import DatasetSplit
    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import Scene, TrajectorySchema
    from dronalize.processing.loading.loader import LoadedSourceData, MapBinding, Source
    from dronalize.processing.models import LoaderRequest
    from dronalize.processing.pipeline.pipeline import Pipeline


_LoaderOptionsT = TypeVar("_LoaderOptionsT", bound=DatasetOptionsModel, default=NoDatasetOptions)


class BaseSceneLoader(ABC, Generic[SourceT, _LoaderOptionsT]):
    """Base class for turning raw dataset sources into canonical trajectory frames.

    Parameters
    ----------
    data_root : Path or str
        The root directory of the dataset, which may be used for source
        discovery or as a base path for source data.
    request : LoaderRequest
        The full loader request, which may be used to configure loading behavior
        and is retained for potential use by subclasses.
    resources : DatasetResources, optional
        Optional shared resources for loading, which may be used by loaders that
        need to share expensive resources like maps across multiple sources or
        splits.

    Notes
    -----
    - `SourceT` is the type of the raw source data used by this loader, such as
      a file path or database query.
    - `_LoaderOptionsT` is the type of the dataset-specific options for this
      loader, which must be a subclass of `DatasetOptionsModel` and defaults to
      `NoDatasetOptions` for loaders without dataset-specific options.
    """

    def __init__(
        self,
        *,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        self.root: Path = Path(data_root)
        self.request: LoaderRequest = request
        self.scenes_config: ScenesConfig = request.scenes
        self.screening_config: ScreeningConfig | None = request.screening
        self.split_config: SplitRequest | None = request.split
        self.map_config: MapConfig | None = request.map
        self.native_splits: tuple[DatasetSplit, ...] | None = request.native_splits
        self.resources: DatasetResources = DatasetResources() if resources is None else resources
        options = cast("_LoaderOptionsT", request.dataset)
        self.dataset_config: _LoaderOptionsT = options
        self.loader_options: _LoaderOptionsT = options
        self._pipeline_cache: Pipeline | None = None

    def __init_subclass__(cls) -> None:
        """Validate that subclasses implement at least one source method."""
        if cls is BaseSceneLoader:
            return

        has_discover_override = cls.discover_sources is not BaseSceneLoader.discover_sources
        has_split_override = cls.sources_for_split is not BaseSceneLoader.sources_for_split
        if not (has_discover_override or has_split_override):
            msg = (
                f"{cls.__name__} must override discover_sources() or sources_for_split() "
                "to provide source enumeration logic."
            )
            raise TypeError(msg)

    @classmethod
    def unified_factory(
        cls,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> Self:
        """Construct a concrete loader instance from the unified request interface.

        Parameters
        ----------
        data_root : Path or str
            Root directory or base path for the dataset.
        request : LoaderRequest
            Full loader request containing all generic and dataset-specific
            options.
        resources : DatasetResources, optional
            Optional shared resource container.

        Returns
        -------
        Self
            A concrete instance of the loader class.

        Notes
        -----
        This method exists to provide a consistent construction interface across
        all loaders, which is especially useful when loaders are selected
        dynamically at runtime.

        Subclasses may override this method if they need custom construction
        behavior, precomputation, validation, or dependency injection beyond the
        default initializer.
        """
        _ = resources
        return cls(data_root=data_root, request=request)

    @classmethod
    def unified_runtime_factory(
        cls,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> RuntimeSceneLoader:
        """Construct a runtime-erased wrapper around this loader.

        Parameters
        ----------
        data_root : Path or str
            Root directory or base path for the dataset.
        request : LoaderRequest
            Full loader request containing all generic and dataset-specific
            options.
        resources : DatasetResources, optional
            Optional shared resource container passed through to loader
            construction.

        Returns
        -------
        RuntimeSceneLoader
            A runtime wrapper around the concrete loader instance.

        Notes
        -----
        This factory is intended for orchestration layers that should not need
        to know the loader's concrete generic type parameters. The returned
        wrapper preserves the operational interface while erasing the specific
        source type.
        """
        return RuntimeSceneLoader.wrap(cls.unified_factory(data_root, request, resources))

    @classmethod
    @abstractmethod
    def native_trajectory_schema(cls) -> TrajectorySchema:
        """Return the native trajectory schema emitted by this loader.

        Returns
        -------
        TrajectorySchema
            The schema describing the raw trajectory fields produced before any
            downstream canonicalization, filtering, or transformation pipeline
            stages are applied.

        Notes
        -----
        This schema is used to derive column metadata and configure the
        processing pipeline appropriately. Concrete loaders should report the
        schema that matches what `load_source()` yields, not necessarily the
        final schema after pipeline execution.
        """

    @abstractmethod
    def load_source(self, source: Source[SourceT]) -> Iterable[LoadedSourceData]:
        """Load one dataset source into one or more lazily consumable scene records.

        Parameters
        ----------
        source : Source[SourceT]
            A source descriptor wrapping the dataset-specific raw source object.
            Depending on the loader, this may represent a file, shard, query, or
            any other unit of dataset input.

        Returns
        -------
        Iterable[LoadedSourceData]
            An iterable of loaded source records. Each record typically contains
            lazy frame data and may optionally include source-level metadata such
            as map bindings or split annotations.

        Notes
        -----
        Implementations should treat this as the primary bridge between raw
        dataset storage and the canonical internal scene-loading pipeline.

        The iterable may be lazy. This is often preferable for large datasets,
        since it allows streaming scene materialization rather than requiring the
        entire source to be loaded into memory at once.
        """

    def pipeline(self) -> Pipeline:
        """Build the processing pipeline used for this loader instance.

        Returns
        -------
        Pipeline
            A fully configured pipeline derived from the current request,
            including scene sampling, optional screening, and split handling.

        Notes
        -----
        The pipeline is constructed from the loader's scene and screening
        configuration together with the native trajectory schema reported by the
        loader.

        The pipeline is cached after the first construction to avoid redundant
        work across multiple sources. If the loader's configuration is immutable
        after construction, this cache will ensure the pipeline is only built
        once.

        """
        if self._pipeline_cache is not None:
            return self._pipeline_cache
        plan = PipelinePlan(
            scenes=self.scenes_config, screening=self.screening_config, split=self.split_config
        )
        columns = TrajectoryColumns.from_schema(self.native_trajectory_schema())
        pipeline = spec.trajectory_pipeline(
            spec.lane_change_sampling(plan, columns=columns)
            if self.scenes_config.lane_change is not None
            else spec.standard(plan, columns=columns)
        )
        self._pipeline_cache = pipeline
        return pipeline

    def clear_pipeline_cache(self) -> None:
        """Clear the cached pipeline to force reconstruction on the next access."""
        self._pipeline_cache = None

    def discover_sources(self) -> Iterable[Source[SourceT]]:
        """Discover all available sources for datasets without native split support.

        Returns
        -------
        Iterable[Source[SourceT]]
            An iterable of source descriptors representing every available raw
            source in the dataset.

        Raises
        ------
        NotImplementedError
            If the concrete loader does not implement source discovery.

        Notes
        -----
        Loaders should implement this method when the dataset is not organized
        around predefined native splits, or when the loader prefers direct
        discovery over split-specific enumeration.

        The default implementation is intentionally strict so that subclasses
        must make an explicit choice between implementing this method,
        implementing `sources_for_split()`, or both.
        """  # noqa: DOC202
        msg = f"{type(self).__name__} must implement discover_sources() or sources_for_split()."
        raise NotImplementedError(msg)

    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[SourceT]]:
        """Yield source descriptors for a specific native dataset split.

        Parameters
        ----------
        split : DatasetSplit
            The dataset-native split to enumerate, such as training,
            validation, or test.

        Returns
        -------
        Iterable[Source[SourceT]]
            An iterable of source descriptors belonging to the requested split.

        Raises
        ------
        SplitNotSupportedError
            If the loader does not support the requested native split.

        Notes
        -----
        Loaders should implement this method when the underlying dataset defines
        explicit native splits and source enumeration depends on the requested
        split.
        """  # noqa: DOC202
        raise SplitNotSupportedError(type(self).__name__, split)

    def all_sources(self) -> Iterable[Source[SourceT]]:
        """Yield all available sources across discovery or native splits.

        Yields
        ------
        Source[SourceT]
            Each source visible to this loader. If native splits are configured,
            yielded sources are annotated with their predefined split.

        Notes
        -----
        The behavior depends on whether the loader advertises native splits:

        - If no native splits are configured, this delegates to
          `discover_sources()`.
        - If native splits are available, this iterates each split via
          `sources_for_split()` and annotates the yielded sources with their
          predefined split.

        This method provides a normalized entry point for orchestration code
        that simply needs all sources, regardless of how the dataset organizes
        them internally.
        """
        supported_splits = self.native_splits or ()
        if len(supported_splits) == 0:
            yield from self.discover_sources()
            return
        for split in supported_splits:
            for source in self.sources_for_split(split):
                yield source.with_predefined_split(split)

    def map_resolver(self) -> MapResolver:
        """Return the loader-level map resolver used for scene-to-map lookup.

        Returns
        -------
        MapResolver
            A callable object that resolves a `Scene` to a `MapGraph`, or to
            `None` if no map can be resolved.

        Notes
        -----
        Concrete loaders may override this to provide dataset-specific map
        lookup logic, caching, or binding rules.

        The default implementation returns a resolver that never supplies a map.
        This is appropriate for datasets without map support or loaders that do
        not participate in map enrichment.
        """
        _ = self
        return no_map()

    def resolve_map(self, scene: Scene, map_binding: MapBinding | None = None) -> MapGraph | None:
        """Resolve the map graph associated with a scene.

        Parameters
        ----------
        scene : Scene
            The scene for which a map should be resolved.
        map_binding : MapBinding, optional
            Optional explicit binding information that may be used by custom
            implementations to refine map lookup.

        Returns
        -------
        MapGraph or None
            The resolved map graph for the scene, or `None` if map resolution is
            disabled or unsuccessful.

        Notes
        -----
        This method first checks whether map support is enabled in the request.
        If no map configuration is present, it returns `None` immediately.

        By default, resolution is delegated to the callable returned by
        `map_resolver()`. Subclasses may override either this method or
        `map_resolver()` depending on whether they need per-call logic or only a
        custom resolver object.
        """
        _ = map_binding
        if self.map_config is None:
            return None
        return self.map_resolver()(scene)

    def num_sources(self) -> int | None:
        """Return the total number of sources when that value is cheaply knowable.

        Returns
        -------
        int or None
            The number of sources available to the loader, or `None` if the
            value is unknown, expensive to compute, or inherently dynamic.

        Notes
        -----
        This method is primarily useful for progress reporting, scheduling, and
        diagnostics. Implementations should only return an integer when doing so
        does not require materializing or exhaustively traversing the full
        source iterator unless that cost is acceptable.
        """
        _ = self
        return None


class RuntimeSceneLoader:
    """Erased runtime wrapper around a concrete typed scene loader.

    Parameters
    ----------
    loader : object
        The concrete loader instance to wrap.

    Notes
    -----
    This wrapper erases the loader's concrete source type so orchestration
    code can interact with heterogeneous loaders through a common runtime
    interface.
    """

    def __init__(self, loader: object) -> None:
        self._loader: object = loader

    @classmethod
    def wrap(cls, loader: object) -> RuntimeSceneLoader:
        """Create a runtime wrapper for a concrete loader instance.

        Parameters
        ----------
        loader : object
            The concrete typed loader to erase.

        Returns
        -------
        RuntimeSceneLoader
            A wrapper exposing the common runtime loading interface.
        """
        return cls(loader)

    @property
    def screening_config(self) -> ScreeningConfig | None:
        """Return the screening configuration associated with the wrapped loader.

        Returns
        -------
        ScreeningConfig or None
            The active screening configuration, if any.
        """
        return self._typed_loader().screening_config

    @property
    def map_config(self) -> MapConfig | None:
        """Return the map configuration associated with the wrapped loader.

        Returns
        -------
        MapConfig or None
            The active map configuration, if any.
        """
        return self._typed_loader().map_config

    def pipeline(self) -> Pipeline:
        """Return the processing pipeline of the wrapped loader.

        Returns
        -------
        Pipeline
            The pipeline configured by the underlying loader.
        """
        return self._typed_loader().pipeline()

    def clear_pipeline_cache(self) -> None:
        """Clear the wrapped loader's cached pipeline to force reconstruction."""
        self._typed_loader().clear_pipeline_cache()

    def discover_sources(self) -> Iterable[Source[Any]]:
        """Discover all sources from the wrapped loader.

        Returns
        -------
        Iterable[Source[Any]]
            An iterable of runtime-erased source descriptors.

        Notes
        -----
        This simply forwards to the underlying loader while erasing the concrete
        source payload type.
        """
        return self._typed_loader().discover_sources()

    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[Any]]:
        """Return all sources for a given split from the wrapped loader.

        Parameters
        ----------
        split : DatasetSplit
            Native dataset split to enumerate.

        Returns
        -------
        Iterable[Source[Any]]
            An iterable of runtime-erased source descriptors for the requested
            split.
        """
        return self._typed_loader().sources_for_split(split)

    def load_source(self, source: Source[Any]) -> Iterable[LoadedSourceData]:
        """Load one source through the wrapped loader.

        Parameters
        ----------
        source : Source[Any]
            Runtime-erased source descriptor compatible with the wrapped loader.

        Returns
        -------
        Iterable[LoadedSourceData]
            Loaded scene data yielded by the underlying loader.

        Notes
        -----
        Type erasure means correctness here depends on the caller providing a
        source that actually matches the wrapped loader's expected source type.
        """
        return self._typed_loader().load_source(source)

    def resolve_map(self, scene: Scene, map_binding: MapBinding | None = None) -> MapGraph | None:
        """Resolve a map for a scene through the wrapped loader.

        Parameters
        ----------
        scene : Scene
            Scene for which map data should be resolved.
        map_binding : MapBinding, optional
            Optional explicit binding information for map resolution.

        Returns
        -------
        MapGraph or None
            The resolved map graph, or `None` when unavailable.
        """
        return self._typed_loader().resolve_map(scene, map_binding)

    def num_sources(self) -> int | None:
        """Return the number of sources reported by the wrapped loader.

        Returns
        -------
        int or None
            Total source count if known, otherwise `None`.
        """
        return self._typed_loader().num_sources()

    def _typed_loader(self) -> BaseSceneLoader[Any, DatasetOptionsModel]:
        return cast("BaseSceneLoader[Any, DatasetOptionsModel]", self._loader)
