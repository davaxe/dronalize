"""Registry and descriptor models for built-in and custom datasets."""

from __future__ import annotations

import functools
import importlib
import importlib.util
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from enum import IntFlag, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from dronalize.core.errors import (
    DatasetNotFoundError,
    DatasetRegistryError,
    MissingOptionalDependencyError,
)
from dronalize.processing.loading.base import NoLoaderOptions
from dronalize.processing.loading.config import LoaderConfig
from dronalize.processing.maps.config import MapConfig

if TYPE_CHECKING:
    from dronalize.core.categories import DatasetSplit
    from dronalize.core.scene.schema import TrajectorySchema
    from dronalize.processing.loading.base import BaseSceneLoader, LoaderOptions
    from dronalize.processing.loading.splits import SplitConfig, SplitStrategyName

_REGISTRY: dict[str, DatasetSpec] = {}

RuntimeContext = Callable[[Path, LoaderConfig, MapConfig | None], AbstractContextManager[None]]


class LoaderFactory(Protocol):
    """Protocol for functions that create dataset loaders with flexible arguments."""

    def __call__(
        self,
        path: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
        loader_options: LoaderOptions | None = None,
    ) -> BaseSceneLoader[Any]:
        """Create a scene loader for the dataset with the given configuration."""
        ...


class DatasetCapabilities(IntFlag):
    """Optional flags to indicate special capabilities."""

    MAP_AVAILABLE = auto()
    """Have map data available that can be included in the output."""
    NATIVE_SPLITS = auto()
    """Expose native dataset splits that can be requested by name."""
    CUSTOM_SPLITS = auto()
    """Support custom split strategies defined by the loader."""
    LANE_CHANGE_SAMPLING = auto()
    """Expose lane-change sampling controls for this dataset."""
    EXECUTION_SCOPE = auto()
    """Require a dataset-specific runtime context to manage shared resources."""
    LOADER_OPTIONS = auto()
    """Loader can accept extra arguments for dataset-specific configuration."""


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Everything needed to fully process a single dataset.

    `DatasetSpec` is the public descriptor type used by the registry and the
    runtime planner. It connects a dataset key to the loader factory, default
    configs, native schema, optional runtime context, and split capability
    metadata that the rest of the library needs in order to plan and execute a
    processing run.
    """

    name: str
    """Canonical slug, e.g. "ind", "argoverse2", or "waymo"."""

    loader_factory: LoaderFactory
    """Factory function that creates a scene loader for the dataset."""

    default_loader_config: LoaderConfig
    """Default loader configuration for the dataset."""

    default_map_config: MapConfig | None
    """Default map configuration for the dataset."""

    native_schema: TrajectorySchema
    """Native trajectory schema for the dataset."""

    loader_options_type: type[LoaderOptions]
    """Typed dataset-specific loader options accepted by the loader factory."""

    default_loader_options: LoaderOptions
    """Default dataset-specific loader options for this dataset."""

    runtime_context_fn: RuntimeContext | None = None
    """Optional runtime context manager factory for dataset-specific resources."""

    predefined_splits: list[DatasetSplit] = field(default_factory=list)
    """Predefined splits exposed by the dataset, if any."""

    supported_split_strategies: list[SplitStrategyName] = field(default_factory=list)
    """Custom split strategies exposed by the loader, if any."""

    recommended_split_strategy: SplitStrategyName | None = None
    """Preferred custom split strategy for automatic selection, if any."""

    capabilities: DatasetCapabilities = field(default=DatasetCapabilities(0))
    """Bitfield of flags indicating special dataset capabilities."""

    @property
    def has_map(self) -> bool:
        """Return whether the dataset exposes map data."""
        return bool(self.capabilities & DatasetCapabilities.MAP_AVAILABLE)

    @property
    def has_runtime_context(self) -> bool:
        """Return whether dataset processing opens a dataset-specific runtime context."""
        return self.runtime_context_fn is not None

    @classmethod
    def from_loader(
        cls,
        name: str,
        loader_cls: type[BaseSceneLoader[Any, Any]],
        loader_factory: LoaderFactory | None = None,
        *,
        runtime_context_fn: RuntimeContext | None = None,
        capabilities: DatasetCapabilities | None = None,
        infer_capabilities: bool = True,
    ) -> DatasetSpec:
        """Create a dataset spec directly from a loader class.

        This is the main helper for custom dataset integrations. It derives the
        default config, native schema, split metadata, and loader-options model
        from the loader class so callers only need to provide the dataset name
        and any optional runtime context or capability overrides.
        """
        resolved_factory = loader_cls if loader_factory is None else loader_factory
        descriptor = cls(
            name=name,
            loader_factory=cast("LoaderFactory", resolved_factory),
            default_loader_config=loader_cls.default_config(),
            default_map_config=loader_cls.default_map_config(),
            native_schema=loader_cls.native_trajectory_schema(),
            loader_options_type=loader_cls.loader_options_model(),
            default_loader_options=loader_cls.default_loader_options(),
            runtime_context_fn=runtime_context_fn,
            predefined_splits=list(loader_cls.predefined_splits()),
            supported_split_strategies=list(loader_cls.supported_split_strategies()),
            recommended_split_strategy=loader_cls.recommended_split_strategy(),
            capabilities=capabilities or DatasetCapabilities(0),
        )
        return descriptor.infer_capabilities() if infer_capabilities else descriptor

    @contextmanager
    def runtime_context(
        self, root: Path, loader_config: LoaderConfig, map_config: MapConfig | None
    ) -> Generator[None, None, None]:
        """Open the dataset-specific runtime context, if one is defined."""
        if self.runtime_context_fn is not None:
            with self.runtime_context_fn(root, loader_config, map_config):
                yield
        else:
            yield

    def with_capabilities(self, *flags: DatasetCapabilities) -> DatasetSpec:
        """Return a copy of this dataset spec with additional capabilities."""
        combined = self.capabilities
        for flag in flags:
            combined |= flag
        return replace(self, capabilities=combined)

    def infer_capabilities(self) -> DatasetSpec:
        """Return a copy of this dataset spec with inferred capabilities based on its properties."""
        flags = DatasetCapabilities(0)
        if self.default_map_config is not None:
            flags |= DatasetCapabilities.MAP_AVAILABLE
        if self.predefined_splits:
            flags |= DatasetCapabilities.NATIVE_SPLITS
        if self.supported_split_strategies or self.recommended_split_strategy is not None:
            flags |= DatasetCapabilities.CUSTOM_SPLITS
        if self.runtime_context_fn is not None:
            flags |= DatasetCapabilities.EXECUTION_SCOPE
        if self.loader_options_type is not NoLoaderOptions:
            flags |= DatasetCapabilities.LOADER_OPTIONS

        return self.with_capabilities(flags)

    def extend_inferred_capabilities(self, *flags: DatasetCapabilities) -> DatasetSpec:
        """Return a copy of this dataset spec with inferred capabilities extended by *flags*."""
        inferred = self.infer_capabilities()
        return inferred.with_capabilities(*flags)

    def build_loader(
        self,
        root: Path,
        *,
        loader_config: LoaderConfig,
        map_config: MapConfig | None,
        trajectory_schema: TrajectorySchema | None,
        loader_options: LoaderOptions,
        splits: Sequence[DatasetSplit] | None = None,
        split_request: SplitConfig | None = None,
    ) -> BaseSceneLoader[Any]:
        """Instantiate the dataset loader with the resolved runtime configuration."""
        if self.loader_options_type is NoLoaderOptions:
            loader = self.loader_factory(
                root,
                loader_config,
                map_config,
                split_request=split_request,
                splits=splits,
            )
        else:
            loader = self.loader_factory(
                root,
                loader_config,
                map_config,
                split_request=split_request,
                splits=splits,
                loader_options=loader_options,
            )
        loader.set_trajectory_schema(trajectory_schema)
        return loader


@dataclass(frozen=True, slots=True)
class _BuiltinDatasetSpec:
    """Lazy import metadata for a built-in dataset."""

    module: str
    export_name: str = "DATASET_SPEC"
    export_key: str | None = None
    optional_dependencies: tuple[str, ...] = ()
    extra: str | None = None


def _builtin(
    module: str,
    *,
    export_name: str = "DATASET_SPEC",
    export_key: str | None = None,
    optional_dependencies: tuple[str, ...] = (),
    extra: str | None = None,
) -> _BuiltinDatasetSpec:
    return _BuiltinDatasetSpec(
        module=module,
        export_name=export_name,
        export_key=export_key,
        optional_dependencies=optional_dependencies,
        extra=extra,
    )


_BUILTIN_DATASETS: dict[str, _BuiltinDatasetSpec] = {
    "a43": _builtin("dronalize.datasets.a43"),
    "ad4che": _builtin("dronalize.datasets.ad4che", optional_dependencies=("cv2",), extra="ad4che"),
    "apolloscape": _builtin("dronalize.datasets.apolloscape"),
    "argoverse1": _builtin("dronalize.datasets.argoverse1"),
    "argoverse2": _builtin("dronalize.datasets.argoverse2"),
    "eth": _builtin("dronalize.datasets.eth_ucy", export_name="DATASET_SPECS", export_key="eth"),
    "hotel": _builtin(
        "dronalize.datasets.eth_ucy", export_name="DATASET_SPECS", export_key="hotel"
    ),
    "univ": _builtin("dronalize.datasets.eth_ucy", export_name="DATASET_SPECS", export_key="univ"),
    "zara1": _builtin(
        "dronalize.datasets.eth_ucy", export_name="DATASET_SPECS", export_key="zara1"
    ),
    "zara2": _builtin(
        "dronalize.datasets.eth_ucy", export_name="DATASET_SPECS", export_key="zara2"
    ),
    "exid": _builtin("dronalize.datasets.exid"),
    "highd": _builtin("dronalize.datasets.highd"),
    "i80": _builtin("dronalize.datasets.i80"),
    "ind": _builtin("dronalize.datasets.ind"),
    "interact": _builtin("dronalize.datasets.interact"),
    "lyft": _builtin(
        "dronalize.datasets.lyft",
        optional_dependencies=("zarr", "numcodecs", "google.protobuf"),
        extra="lyft",
    ),
    "nuscenes": _builtin("dronalize.datasets.nuscenes"),
    "opendd": _builtin("dronalize.datasets.opendd"),
    "round": _builtin("dronalize.datasets.round"),
    "sind": _builtin("dronalize.datasets.sind"),
    "unid": _builtin("dronalize.datasets.unid"),
    "us101": _builtin("dronalize.datasets.us101"),
    "vod": _builtin("dronalize.datasets.vod"),
    "waymo": _builtin(
        "dronalize.datasets.waymo", optional_dependencies=("google.protobuf",), extra="waymo"
    ),
}


def register(descriptor: DatasetSpec) -> None:
    """Register a dataset descriptor for later lookup by name.

    This is the public entry point for custom dataset integrations that want to
    participate in the normal planning and runtime flow.
    """
    if descriptor.name in _REGISTRY and _REGISTRY[descriptor.name] != descriptor:
        msg = f"Dataset '{descriptor.name}' is already registered."
        raise DatasetRegistryError(msg)

    _REGISTRY[descriptor.name] = descriptor


def get(name: str) -> DatasetSpec:
    """Get a dataset spec by name."""
    if name in _REGISTRY:
        return _REGISTRY[name]

    builtin_specs = _builtin_datasets()
    if name not in builtin_specs:
        raise DatasetNotFoundError(name, available())

    spec = builtin_specs[name]
    missing = _missing_optional_dependencies(spec)
    if missing:
        raise _missing_dependency_error(subject=f"Dataset '{name}'", spec=spec, missing=missing)

    return _load_builtin_descriptor(name)


def available() -> list[str]:
    """Get a list of available dataset names."""
    builtin_names = {
        name
        for name, spec in _builtin_datasets().items()
        if not _missing_optional_dependencies(spec)
    }
    return sorted(set(_REGISTRY) | builtin_names)


def _builtin_datasets() -> dict[str, _BuiltinDatasetSpec]:
    """Return the static built-in dataset table."""
    return _BUILTIN_DATASETS


@functools.cache
def _load_builtin_descriptor(name: str) -> DatasetSpec:
    """Import and resolve a built-in dataset spec by dataset name."""
    spec = _builtin_datasets().get(name)
    if spec is None:
        raise DatasetNotFoundError(name, available())

    module = importlib.import_module(spec.module)

    try:
        exported = getattr(module, spec.export_name)
    except AttributeError as exc:
        msg = (
            f"Built-in dataset module '{spec.module}' does not export '{spec.export_name}' "
            f"for dataset '{name}'."
        )
        raise DatasetRegistryError(msg) from exc

    descriptor = exported
    if spec.export_key is not None:
        try:
            descriptor = exported[spec.export_key]
        except (KeyError, TypeError) as exc:
            msg = (
                f"Built-in dataset module '{spec.module}' does not define descriptor "
                f"'{spec.export_key}' in '{spec.export_name}'."
            )
            raise DatasetRegistryError(msg) from exc

    if not isinstance(descriptor, DatasetSpec):
        msg = f"Built-in dataset '{name}' did not resolve to a DatasetSpec."
        raise DatasetRegistryError(msg)

    if descriptor.name != name:
        msg = (
            f"Built-in dataset '{name}' resolved to descriptor '{descriptor.name}'. "
            "Descriptor names must match builtin registry keys."
        )
        raise DatasetRegistryError(msg)

    return descriptor


@functools.lru_cache
def _missing_optional_dependencies(spec: _BuiltinDatasetSpec) -> tuple[str, ...]:
    """List optional dependencies that are unavailable for a built-in dataset."""
    return tuple(
        module_name for module_name in spec.optional_dependencies if not _has_module(module_name)
    )


@functools.lru_cache
def _has_module(module_name: str) -> bool:
    """Return whether *module_name* can be imported in the current environment."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _missing_dependency_error(
    *, subject: str, spec: _BuiltinDatasetSpec, missing: tuple[str, ...]
) -> MissingOptionalDependencyError:
    """Build a consistent error for unavailable optional dataset dependencies."""
    install_target = f"dronalize[{spec.extra}]" if spec.extra else None
    install_hint = f"Install {install_target} to use it." if install_target else ""
    missing_str = ", ".join(missing)
    msg = (
        f"{subject} is unavailable because the following optional dependencies are missing: "
        f"{missing_str}. {install_hint}"
    )
    return MissingOptionalDependencyError(
        msg, dependencies=tuple(missing), install_target=install_target
    )
