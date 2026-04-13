"""Base loader implementation for thin dataset-ingestion adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Generic, cast

from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import Self, TypeVar

from dronalize.core.errors import LoaderConfigError, SplitNotSupportedError
from dronalize.core.typing import SourceT
from dronalize.processing.loading.resources import EMPTY_DATASET_RESOURCES, DatasetResources
from dronalize.processing.maps.resolver import MapResolver, no_map
from dronalize.processing.models import PipelinePlan, SplitRequest
from dronalize.processing.pipeline import spec
from dronalize.processing.pipeline.factory import trajectory_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.config.models import MapConfig, ScenesConfig, ScreeningConfig
    from dronalize.core.categories import DatasetSplit
    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import Scene, TrajectorySchema
    from dronalize.processing.loading.loader import LoadedSourceData, MapBinding, Source
    from dronalize.processing.models import LoaderRequest
    from dronalize.processing.pipeline.pipeline import Pipeline


class DatasetOptionsModel(BaseModel):
    """Base model for dataset-specific declarative config."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class NoDatasetOptions(DatasetOptionsModel):
    """Empty dataset config for datasets without dataset-owned settings."""


LoaderOptions = DatasetOptionsModel
NoLoaderOptions = NoDatasetOptions


@dataclass(slots=True, frozen=True, kw_only=True)
class LoaderSplitCapabilities:
    """Legacy marker kept only until the dataset library is migrated."""

    supports_source_split: bool = False
    supports_scene_split: bool = False
    supports_block_split: bool = False


_LoaderOptionsT = TypeVar("_LoaderOptionsT", bound=DatasetOptionsModel, default=NoDatasetOptions)


class BaseSceneLoader(ABC, Generic[SourceT, _LoaderOptionsT]):
    """Base class for turning raw dataset sources into canonical trajectory frames."""

    def __init__(
        self,
        *,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        """Initialize one loader from a narrow loader-facing request."""
        self.root: Path = Path(data_root)
        self.request: LoaderRequest = request
        self.scenes_config: ScenesConfig = request.scenes
        self.screening_config: ScreeningConfig | None = request.screening
        self.split_config: SplitRequest | None = request.split
        self.map_config: MapConfig | None = request.map
        self.native_splits: tuple[DatasetSplit, ...] | None = request.native_splits
        self.resources: DatasetResources = (
            EMPTY_DATASET_RESOURCES if resources is None else resources
        )
        options = self.coerce_loader_options(request.dataset)
        self.dataset_config: _LoaderOptionsT = options
        self.loader_options: _LoaderOptionsT = options

    @classmethod
    def loader_options_model(cls) -> type[_LoaderOptionsT]:
        """Return the options model used by this loader."""
        return cast("type[_LoaderOptionsT]", NoDatasetOptions)

    @classmethod
    def default_loader_options(cls) -> _LoaderOptionsT:
        """Return the default dataset-specific options for this loader."""
        return cls.loader_options_model()()

    @classmethod
    def parse_loader_options(cls, options_dict: Mapping[str, object] | None) -> _LoaderOptionsT:
        """Parse typed loader options from a plain mapping."""
        try:
            return cls.loader_options_model().model_validate(options_dict or {})
        except ValidationError as exc:
            msg = f"Invalid loader options for {cls.__name__}: {exc}"
            raise LoaderConfigError(msg) from exc

    @classmethod
    def coerce_loader_options(cls, payload: object | None) -> _LoaderOptionsT:
        """Return typed loader options from a mapping, model instance, or `None`."""
        if payload is None:
            return cls.default_loader_options()

        options_model = cls.loader_options_model()
        if isinstance(payload, options_model):
            return payload
        if isinstance(payload, BaseModel):
            if isinstance(payload, DatasetOptionsModel) and options_model is NoDatasetOptions:
                return cast("_LoaderOptionsT", payload)
            return cls.parse_loader_options(payload.model_dump())
        if isinstance(payload, Mapping):
            return cls.parse_loader_options(cast("Mapping[str, object]", payload))

        msg = f"Invalid loader options payload for {cls.__name__}: {type(payload).__name__}"
        raise LoaderConfigError(msg)

    @classmethod
    def unified_factory(
        cls,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> Self:
        """Unified factory method for constructing this loader from a loader request."""
        _ = resources  # ingore because most loaders dont use resources
        return cls(data_root=data_root, request=request)

    @classmethod
    @abstractmethod
    def native_trajectory_schema(cls) -> TrajectorySchema:
        """Return the native schema produced by this loader."""

    @abstractmethod
    def load_source(self, source: Source[SourceT]) -> Iterable[LoadedSourceData]:
        """Read one source into lazy frames plus optional map metadata."""

    def pipeline(self) -> Pipeline:
        """Return the processing pipeline for this loader."""
        plan = PipelinePlan(
            scenes=self.scenes_config,
            screening=self.screening_config,
            split=self.split_config,
        )
        return trajectory_pipeline(
            spec.lane_change_sampling(plan)
            if self.scenes_config.lane_change is not None
            else spec.standard(plan),
        )

    def discover_sources(self) -> Iterable[Source[SourceT]]:
        """Discover all available sources for datasets without native splits."""
        msg = f"{type(self).__name__} must implement discover_sources() or sources_for_split()."
        raise NotImplementedError(msg)

    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[SourceT]]:
        """Yield sources for one native dataset split."""
        raise SplitNotSupportedError(type(self).__name__, split)

    def all_sources(self) -> Iterable[Source[SourceT]]:
        """Yield all sources across all native splits or discovery paths."""
        supported_splits = self.native_splits or ()
        if len(supported_splits) == 0:
            yield from self.discover_sources()
            return
        for split in supported_splits:
            for source in self.sources_for_split(split):
                yield source.with_predefined_split(split)

    def map_resolver(self) -> MapResolver:
        """Return the loader-level map resolver."""
        _ = self
        return no_map()

    def resolve_map(self, scene: Scene, map_binding: MapBinding | None = None) -> MapGraph | None:
        """Resolve one scene map from the loader's configured map backend."""
        _ = map_binding
        if self.map_config is None:
            return None
        return self.map_resolver()(scene)

    def num_sources(self) -> int | None:
        """Return the number of sources when it can be known ahead of time."""
        _ = self
        return None
