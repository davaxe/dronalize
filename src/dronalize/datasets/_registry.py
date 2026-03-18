"""Registry for dataset descriptors."""

from __future__ import annotations

import functools
import importlib
import importlib.util
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate

from dronalize._internal._types import P
from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.exceptions import (
    DatasetNotFoundError,
    DatasetRegistryError,
    MissingOptionalDependencyError,
)
from dronalize.loading import BaseSceneLoader

if TYPE_CHECKING:
    from dronalize.categories import DatasetSplit

_REGISTRY: dict[str, DatasetDescriptor] = {}

ExecutionScope = Callable[[Path, LoaderConfig, MapConfig], AbstractContextManager[None]]
"""Function for creating an execution context for a dataset."""

LoaderFactory = Callable[
    Concatenate[Path | str, LoaderConfig | None, MapConfig | None, P], BaseSceneLoader[Any]
]


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Everything needed to fully process a single dataset."""

    name: str
    """Canonical slug, e.g. "ind", "argoverse2", or "waymo"."""

    loader_factory: LoaderFactory[...]
    """Factory function that creates a scene loader for the dataset."""

    default_config: LoaderConfig
    """Default loader configuration for the dataset."""

    default_map_config: MapConfig
    """Default map configuration for the dataset, if applicable."""

    execution_scope_fn: ExecutionScope | None = None
    """Optional runtime context manager factory for dataset-specific resources."""

    has_map: bool = False
    """Whether the dataset exposes map data."""

    predefined_splits: list[DatasetSplit] = field(default_factory=list)
    """Predefined splits exposed by the dataset, if any."""

    @contextmanager
    def execution_scope(
        self, root: Path, loader_config: LoaderConfig, map_config: MapConfig
    ) -> Generator[None, None, None]:
        """Execute the lifecycle context manager, if defined."""
        if self.execution_scope_fn is not None:
            with self.execution_scope_fn(root, loader_config, map_config):
                yield
        else:
            yield


@dataclass(frozen=True, slots=True)
class _BuiltinDatasetSpec:
    """Lazy import metadata for a built-in dataset."""

    module: str
    export_name: str = "DESCRIPTOR"
    export_key: str | None = None
    optional_dependencies: tuple[str, ...] = ()
    extra: str | None = None


def _builtin(
    module: str,
    *,
    export_name: str = "DESCRIPTOR",
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
    "ad4che": _builtin(
        "dronalize.datasets.ad4che",
        optional_dependencies=("cv2",),
        extra="ad4che",
    ),
    "apolloscape": _builtin("dronalize.datasets.apolloscape"),
    "argoverse1": _builtin("dronalize.datasets.argoverse1"),
    "argoverse2": _builtin("dronalize.datasets.argoverse2"),
    "eth": _builtin("dronalize.datasets.eth_ucy", export_name="DESCRIPTORS", export_key="eth"),
    "hotel": _builtin(
        "dronalize.datasets.eth_ucy",
        export_name="DESCRIPTORS",
        export_key="hotel",
    ),
    "univ": _builtin("dronalize.datasets.eth_ucy", export_name="DESCRIPTORS", export_key="univ"),
    "zara1": _builtin(
        "dronalize.datasets.eth_ucy",
        export_name="DESCRIPTORS",
        export_key="zara1",
    ),
    "zara2": _builtin(
        "dronalize.datasets.eth_ucy",
        export_name="DESCRIPTORS",
        export_key="zara2",
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
        "dronalize.datasets.waymo",
        optional_dependencies=("google.protobuf",),
        extra="waymo",
    ),
}


def register(descriptor: DatasetDescriptor) -> None:
    """Register a dataset descriptor."""
    if descriptor.name in _REGISTRY and _REGISTRY[descriptor.name] != descriptor:
        msg = f"Dataset '{descriptor.name}' is already registered."
        raise DatasetRegistryError(msg)

    _REGISTRY[descriptor.name] = descriptor


def get(name: str) -> DatasetDescriptor:
    """Get a dataset descriptor by name."""
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
def _load_builtin_descriptor(name: str) -> DatasetDescriptor:
    """Import and resolve a built-in descriptor by dataset name."""
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

    if not isinstance(descriptor, DatasetDescriptor):
        msg = f"Built-in dataset '{name}' did not resolve to a DatasetDescriptor."
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
    *,
    subject: str,
    spec: _BuiltinDatasetSpec,
    missing: tuple[str, ...],
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
        msg,
        dependencies=tuple(missing),
        install_target=install_target,
    )
