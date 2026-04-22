"""Base loader implementation for thin dataset-ingestion adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Generic, cast

from typing_extensions import Self, TypeVar

from dronalize.core.errors import SplitNotSupportedError
from dronalize.core.typing import SourceT
from dronalize.processing.loading.options import DatasetOptionsModel, NoDatasetOptions
from dronalize.processing.loading.resources import DatasetResources
from dronalize.processing.maps.resolver import MapResolver, no_map

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.config.models import MapConfig, ScenesConfig, ScreeningConfig
    from dronalize.core.categories import DatasetSplit
    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import Scene, TrajectorySchema
    from dronalize.processing.loading.loader import LoadedSourceData, MapBinding, Source
    from dronalize.processing.models import LoaderRequest, SplitRequest


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
        self.split_config: SplitRequest = request.split
        self.map_config: MapConfig | None = request.map
        self.native_splits: tuple[DatasetSplit, ...] | None = request.native_splits
        self.resources: DatasetResources = DatasetResources() if resources is None else resources
        options = cast("_LoaderOptionsT", request.dataset)
        self.dataset_config: _LoaderOptionsT = options
        self.loader_options: _LoaderOptionsT = options

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
