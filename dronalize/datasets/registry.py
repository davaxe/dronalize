"""Registry for dataset descriptors."""

from __future__ import annotations

import importlib
import importlib.util
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from enum import IntEnum, auto
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import tomllib

from dronalize.core.datatypes.map_config import MapConfig
from dronalize.core.datatypes.split import DatasetSplit

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from dronalize.core.datatypes.loader_config import LoaderConfig
    from dronalize.core.protocols.loader import BaseSceneLoader

# ==============================================================================
# Constants & Global State
# ==============================================================================

_MANIFEST_NAME = "manifest.toml"
_REGISTRY: dict[str, DatasetDescriptor] = {}


# ==============================================================================
# Protocols, Enums & Dataclasses
# ==============================================================================


class DatasetLifecycleContext(Protocol):
    """Defines the resource lifecycle contract for a dataset.

    Implementations must act as context managers that perform necessary
    initialization (e.g., allocating shared memory for maps) before yielding,
    and guarantee resource cleanup after the context exits.
    """

    def __call__(
        self, root: Path, loader_config: LoaderConfig, map_config: MapConfig
    ) -> AbstractContextManager[None]:
        """Execute the dataset lifecycle context.

        Parameters
        ----------
        root : Path
            Root directory of the dataset.
        loader_config : LoaderConfig
            Configuration for the dataset loader.
        map_config : MapConfig
            Configuration for building or loading the map graph.

        Returns
        -------
        AbstractContextManager[None]
            A context manager governing the dataset's temporary resources.
        """
        ...


class MapMode(IntEnum):
    """How a dataset exposes map data at runtime."""

    NONE = auto()
    BUILDER_ONLY = auto()
    INLINE = auto()
    LAZY_KEYED = auto()
    SHARED_SINGLE = auto()
    SHARED_KEYED = auto()


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Everything needed to fully process a single dataset."""

    name: str
    """Canonical slug, e.g. "ind", "argoverse2", "waymo"."""

    loader_factory: Callable[..., BaseSceneLoader]
    """Factory function that creates a scene loader for the dataset."""

    default_config: LoaderConfig
    """Default loader configuration for the dataset."""

    default_map_config: MapConfig = field(default_factory=MapConfig.default)
    """Default map configuration for the dataset, if applicable."""

    lifecycle_context: DatasetLifecycleContext | None = None
    """Optional lifecycle context for the dataset, which can manage resources like shared memory."""

    map_mode: MapMode = MapMode.NONE
    """How this dataset exposes map data at runtime."""

    predefined_splits: list[DatasetSplit] | None = None
    """Predefined splits for the dataset, if any."""

    @property
    def has_map(self) -> bool:
        """Compatibility flag for code that only distinguishes map vs. no map."""
        return self.map_mode is not MapMode.NONE

    def with_splits(
        self, *splits: DatasetSplit | Literal["train", "val", "test"]
    ) -> DatasetDescriptor:
        """Return a copy of this descriptor with the specified predefined splits."""
        normalized_splits = [DatasetSplit(s) if isinstance(s, str) else s for s in splits]
        return replace(self, predefined_splits=normalized_splits)

    def with_all_splits(self) -> DatasetDescriptor:
        """Indicate that this dataset has all three standard splits (train, val, test)."""
        return self.with_splits(DatasetSplit.TEST, DatasetSplit.TRAIN, DatasetSplit.VAL)

    @contextmanager
    def execute_lifecycle_context(
        self, root: Path, loader_config: LoaderConfig, map_config: MapConfig
    ) -> Generator[None, None, None]:
        """Execute the lifecycle context manager, if defined."""
        if self.lifecycle_context is not None:
            with self.lifecycle_context(root, loader_config, map_config):
                yield
        else:
            yield


@dataclass(frozen=True, slots=True)
class _BuiltinDatasetSpec:
    """Lazy import metadata for a built-in dataset."""

    module: str
    optional_dependencies: tuple[str, ...] = ()
    extra: str | None = None


# ==============================================================================
# Public API
# ==============================================================================


def register(descriptor: DatasetDescriptor) -> DatasetDescriptor:
    """Register a dataset descriptor.

    Parameters
    ----------
    descriptor : DatasetDescriptor
        The descriptor to register.

    Raises
    ------
    ValueError
        If a dataset with the same canonical name has already been registered.
    """
    if descriptor.name in _REGISTRY and _REGISTRY[descriptor.name] != descriptor:
        msg = f"Dataset '{descriptor.name}' is already registered."
        raise ValueError(msg)

    _REGISTRY[descriptor.name] = descriptor
    return descriptor


def get(name: str) -> DatasetDescriptor:
    """Get a dataset descriptor by name.

    Parameters
    ----------
    name : str
        The name of the dataset to retrieve.

    Returns
    -------
    DatasetDescriptor
        The descriptor for the requested dataset.

    Raises
    ------
    KeyError
        If the dataset name is unknown.
    """
    _ensure_registered(name)

    if name not in _REGISTRY:
        known = ", ".join(available()) or "none"
        msg = f"Unknown dataset '{name}'. Available datasets: {known}."
        raise KeyError(msg)

    return _REGISTRY[name]


def available() -> list[str]:
    """Get a list of available dataset names.

    Returns
    -------
    list[str]
        A sorted list of registered datasets plus built-in datasets whose
        optional dependencies are currently installed.
    """
    builtin_names = {
        name
        for name, spec in _builtin_datasets().items()
        if not _missing_optional_dependencies(spec)
    }
    return sorted(set(_REGISTRY) | builtin_names)


# ==============================================================================
# Internal Helpers
# ==============================================================================


def _parse_manifest(manifest_path: Path) -> dict[str, _BuiltinDatasetSpec]:
    """Load one dataset manifest file into per-dataset specs."""
    with manifest_path.open("rb") as manifest_file:
        raw_manifest = tomllib.load(manifest_file)

    module = str(raw_manifest.get("module"))
    dataset_names = raw_manifest.get("datasets", [])
    optional_deps = raw_manifest.get("optional_dependencies", [])
    extra = raw_manifest.get("extra")

    if not isinstance(dataset_names, list) or not all(isinstance(n, str) for n in dataset_names):
        msg = f"Invalid dataset manifest {manifest_path}: 'datasets' must be a list of strings."
        raise ValueError(msg)

    if not isinstance(optional_deps, list) or not all(isinstance(n, str) for n in optional_deps):
        msg = (
            f"Invalid dataset manifest {manifest_path}: "
            f"'optional_dependencies' must be a list of strings."
        )
        raise ValueError(msg)

    if extra is not None and not isinstance(extra, str):
        msg = f"Invalid dataset manifest {manifest_path}: 'extra' must be a string or null."
        raise ValueError(msg)

    spec = _BuiltinDatasetSpec(
        module=module,
        optional_dependencies=tuple(optional_deps),
        extra=extra,
    )
    return dict.fromkeys(dataset_names, spec)


@cache
def _builtin_datasets() -> dict[str, _BuiltinDatasetSpec]:
    """Discover built-in dataset specs from per-package manifest files."""
    datasets_dir = Path(__file__).resolve().parent
    builtin_specs: dict[str, _BuiltinDatasetSpec] = {}

    for manifest_path in datasets_dir.rglob(_MANIFEST_NAME):
        if not manifest_path.is_file():
            continue

        for dataset_name, spec in _parse_manifest(manifest_path).items():
            if dataset_name in builtin_specs and builtin_specs[dataset_name] != spec:
                msg = (
                    f"Dataset '{dataset_name}' is defined more than once across dataset manifests."
                )
                raise ValueError(msg)
            builtin_specs[dataset_name] = spec

    return builtin_specs


def _has_module(module_name: str) -> bool:
    """Return whether *module_name* can be imported in the current environment."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _missing_optional_dependencies(spec: _BuiltinDatasetSpec) -> list[str]:
    """List optional dependencies that are unavailable for a built-in dataset."""
    return [
        module_name for module_name in spec.optional_dependencies if not _has_module(module_name)
    ]


def _ensure_registered(name: str) -> None:
    """Import the built-in dataset module for *name* on first use."""
    if name in _REGISTRY:
        return

    builtin_specs = _builtin_datasets()
    if name not in builtin_specs:
        return

    spec = builtin_specs[name]
    missing = _missing_optional_dependencies(spec)

    if missing:
        install_hint = (
            f" Install the optional extra with `pip install dronalize[{spec.extra}]`."
            if spec.extra
            else ""
        )
        missing_str = ", ".join(missing)
        msg = (
            f"Dataset '{name}' is unavailable because optional dependencies are missing: "
            f"{missing_str}.{install_hint}"
        )
        raise ModuleNotFoundError(msg)

    importlib.import_module(spec.module)
