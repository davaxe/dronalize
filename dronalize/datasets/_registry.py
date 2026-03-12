"""Registry for dataset descriptors."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from enum import IntEnum, auto
from functools import cache
from pathlib import Path
from typing import Any, Concatenate, Literal

from pydantic import BaseModel

from dronalize._internal._types import P
from dronalize.categories import DatasetSplit
from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.loading import BaseSceneLoader

_BUILTIN_METADATA_NAME = "__dronalize_builtin__"
_NON_DATASET_PACKAGES = frozenset({"common"})
_REGISTRY: dict[str, DatasetDescriptor] = {}

ExecutionScope = Callable[[Path, LoaderConfig, MapConfig], AbstractContextManager[None]]
"""Function for creating an execution context for a dataset.

The context manager returned by this function can be used to manage resources
that are needed during the execution of a dataset loader, such as shared memory
for map data.

"""


class MapMode(IntEnum):
    """How a dataset exposes map data at runtime."""

    NONE = auto()
    """The dataset does not include map data."""

    BUILDER_ONLY = auto()
    """No map data is included at runtime, but builder is available."""

    INLINE = auto()
    """Map data is included in the same files as the scene data."""

    LAZY_KEYED = auto()
    """Map data is stored separately from the scene data and accessed via keys.

    These are accessed lazily, meaning that the map data for a scene is only
    loaded when it is explicitly requested by the scene loader.
    """

    SHARED_SINGLE = auto()
    """Map data is built once and stored in shared memory for access."""

    SHARED_KEYED = auto()
    """Similar to SHARED_SINGLE, but supports multiple maps distinguished by keys."""


LoaderFactory = Callable[
    Concatenate[Path | str, LoaderConfig | None, MapConfig | None, P], BaseSceneLoader[Any]
]


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Everything needed to fully process a single dataset."""

    name: str
    """Canonical slug, e.g. "ind", "argoverse2", "waymo"."""

    loader_factory: LoaderFactory[...]
    """Factory function that creates a scene loader for the dataset."""

    default_config: LoaderConfig
    """Default loader configuration for the dataset."""

    default_map_config: MapConfig
    """Default map configuration for the dataset, if applicable."""

    execution_scope_fn: ExecutionScope | None = None
    """Optional runtime context for the dataset, which can manage resources like shared memory."""

    map_mode: MapMode = MapMode.NONE
    """How this dataset exposes map data at runtime."""

    predefined_splits: list[DatasetSplit] = field(default_factory=list)
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
    optional_dependencies: tuple[str, ...] = ()
    extra: str | None = None


class _BuiltinModuleMetadata(BaseModel):
    datasets: list[str]
    optional_dependencies: list[str] | None = None
    extra: str | None = None


def register(descriptor: DatasetDescriptor) -> None:
    """Register a dataset descriptor."""
    if descriptor.name in _REGISTRY and _REGISTRY[descriptor.name] != descriptor:
        msg = f"Dataset '{descriptor.name}' is already registered."
        raise ValueError(msg)

    _REGISTRY[descriptor.name] = descriptor


def get(name: str) -> DatasetDescriptor:
    """Get a dataset descriptor by name."""
    _ensure_registered(name)

    if name not in _REGISTRY:
        known_datasets = ", ".join(available()) or "none"
        msg = f"Unknown dataset '{name}'. Available datasets: {known_datasets}."
        raise KeyError(msg)

    return _REGISTRY[name]


def available() -> list[str]:
    """Get a list of available dataset names."""
    builtin_names = {
        name
        for name, spec in _builtin_datasets().items()
        if not _missing_optional_dependencies(spec)
    }
    return sorted(set(_REGISTRY.keys()) | builtin_names)


def ensure_builtin_dependencies(module_name: str, metadata: dict[str, Any]) -> None:
    """Validate optional dependencies before importing a built-in dataset package."""
    normalized_metadata = _BuiltinModuleMetadata.model_validate(metadata)
    spec = _builtin_spec(module_name, normalized_metadata)
    missing = _missing_optional_dependencies(spec)

    if not missing:
        return

    subject = (
        f"Dataset '{normalized_metadata.datasets[0]}'"
        if len(normalized_metadata.datasets) == 1
        else f"Dataset module '{module_name}'"
    )
    raise _missing_dependency_error(subject=subject, spec=spec, missing=missing)


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
        raise _missing_dependency_error(subject=f"Dataset '{name}'", spec=spec, missing=missing)

    _ = importlib.import_module(spec.module)


@cache
def _builtin_datasets() -> dict[str, _BuiltinDatasetSpec]:
    """Discover built-in dataset specs from dataset package metadata."""
    datasets_dir = Path(__file__).resolve().parent
    builtin_specs: dict[str, _BuiltinDatasetSpec] = {}

    for package_dir in datasets_dir.iterdir():
        if not package_dir.is_dir() or package_dir.name in _NON_DATASET_PACKAGES:
            continue

        package_init = package_dir / "__init__.py"
        if not package_init.is_file():
            continue

        for dataset_name, spec in _parse_builtin_package(package_init).items():
            if dataset_name in builtin_specs and builtin_specs[dataset_name] != spec:
                msg = f"Dataset '{dataset_name}' is defined more than once across dataset packages."
                raise ValueError(msg)
            builtin_specs[dataset_name] = spec

    return builtin_specs


def _parse_builtin_package(package_init: Path) -> dict[str, _BuiltinDatasetSpec]:
    """Load one dataset package into per-dataset lazy import specs."""
    metadata = _extract_builtin_metadata(package_init)
    module_name = f"dronalize.datasets.{package_init.parent.name}"
    spec = _builtin_spec(module_name, metadata)
    return dict.fromkeys(metadata.datasets, spec)


def _extract_builtin_metadata(package_init: Path) -> _BuiltinModuleMetadata:
    """Read built-in dataset metadata from a package `__init__` module."""
    try:
        module_ast = ast.parse(package_init.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        msg = f"Could not parse dataset package metadata from '{package_init}'."
        raise ValueError(msg) from exc

    for node in module_ast.body:
        target_value = None

        # Extract value directly within the narrowed type blocks
        if isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Name) and t.id == _BUILTIN_METADATA_NAME for t in node.targets
            ):
                target_value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == _BUILTIN_METADATA_NAME
        ):
            target_value = node.value

        if target_value is not None:
            try:
                raw_metadata = ast.literal_eval(target_value)
                return _BuiltinModuleMetadata.model_validate(raw_metadata)
            except (ValueError, TypeError, SyntaxError) as exc:
                msg = (
                    f"Could not parse dataset package metadata from '{package_init}'. "
                    f"Ensure that the variable {_BUILTIN_METADATA_NAME!r} "
                    "is defined as a Python literal."
                )
                raise ValueError(msg) from exc

    msg = f"Dataset package '{package_init.parent.name}' must define {_BUILTIN_METADATA_NAME!r}."
    raise ValueError(msg)


def _builtin_spec(module_name: str, metadata: _BuiltinModuleMetadata) -> _BuiltinDatasetSpec:
    """Build a lazy import spec from normalized package metadata."""
    return _BuiltinDatasetSpec(
        module=module_name,
        optional_dependencies=tuple(metadata.optional_dependencies or []),
        extra=metadata.extra,
    )


def _missing_optional_dependencies(spec: _BuiltinDatasetSpec) -> list[str]:
    """List optional dependencies that are unavailable for a built-in dataset."""
    return [
        module_name for module_name in spec.optional_dependencies if not _has_module(module_name)
    ]


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
    missing: list[str],
) -> ModuleNotFoundError:
    """Build a consistent error for unavailable optional dataset dependencies."""
    install_hint = (
        f"Install the optional extra with `pip install dronalize[{spec.extra}]`."
        if spec.extra
        else ""
    )
    missing_str = ", ".join(missing)
    msg = (
        f"{subject} is unavailable because the following optional dependencies are missing: "
        f"{missing_str}. {install_hint}"
    )
    return ModuleNotFoundError(msg)
