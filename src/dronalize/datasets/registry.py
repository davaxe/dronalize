"""Explicit dataset registry and descriptor models."""

from __future__ import annotations

import functools
import importlib
import importlib.util
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import ValidationError

from dronalize.config.models import DatasetConfig, MapConfig, ScenesConfig
from dronalize.core.errors import (
    DatasetNotFoundError,
    DatasetRegistryError,
    LoaderConfigError,
    MissingOptionalDependencyError,
)
from dronalize.processing.loading.options import DatasetOptionsModel, NoDatasetOptions
from dronalize.processing.loading.resources import DatasetResources
from dronalize.processing.models import LoaderRequest, ReadRequest

if TYPE_CHECKING:
    from dronalize.core.categories import DatasetSplit
    from dronalize.core.scene import TrajectorySchema


_REGISTRY: dict[str, DatasetSpec] = {}

ResourcesFactory = Callable[
    [Path, ScenesConfig, MapConfig | None], AbstractContextManager[DatasetResources]
]
"""Factory signature for dataset-scoped shared resources.

A resources factory receives the dataset root plus the resolved scene and map
configuration for a run, then returns a context manager that owns shared state
such as cached metadata tables, shared-memory map stores, or handles reused
across loader instances.
"""


@dataclass(slots=True, frozen=True, kw_only=True)
class DatasetSplitSupport:
    scene: bool = True
    source: bool = False
    time_block: bool = False


class LoaderFactory(Protocol):
    """Protocol for functions that create dataset loaders with flexible arguments."""

    def __call__(
        self,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> object:
        """Create a scene loader for the dataset with the given configuration."""
        ...


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Explicit descriptor for one dataset integration."""

    name: str
    loader_factory: LoaderFactory
    default_config: DatasetConfig
    native_schema: TrajectorySchema
    supported_native_splits: tuple[DatasetSplit, ...] | None = None
    dataset_options_model: type[DatasetOptionsModel] = NoDatasetOptions
    resources_factory: ResourcesFactory | None = None
    has_map: bool = False
    split_support: DatasetSplitSupport = DatasetSplitSupport()

    def default_dataset_options(self) -> DatasetOptionsModel:
        """Return the default typed dataset-owned config block."""
        return self.dataset_options_model()

    def parse_dataset_config(self, payload: Mapping[str, object] | None) -> DatasetOptionsModel:
        """Parse and validate dataset-owned config from plain data."""
        try:
            return self.dataset_options_model.parse(dict(payload or {}))
        except ValidationError as exc:
            msg = f"Invalid dataset config for dataset '{self.name}': {exc}"
            raise LoaderConfigError(msg) from exc

    @contextmanager
    def open_resources(
        self, root: Path, request: LoaderRequest
    ) -> Generator[DatasetResources, None, None]:
        """Open per-run shared dataset resources."""
        if self.resources_factory is None:
            yield DatasetResources()
            return
        with self.resources_factory(root, request.scenes, request.map) as resources:
            yield resources

    def default_loader_request(self) -> LoaderRequest:
        """Return the default loader-facing request for this dataset."""
        return LoaderRequest(
            scenes=self.default_config.scenes,
            screening=self.default_config.screening,
            read=ReadRequest.from_config(
                self.default_config.read, supported_native_splits=self.supported_native_splits
            ),
            dataset=self.default_dataset_options(),
            map=self.default_config.map,
        )

    def build_loader(
        self, *, root: Path, request: LoaderRequest, resources: DatasetResources | None = None
    ) -> object:
        """Construct one loader instance for this dataset specification."""
        return self.loader_factory(data_root=root, request=request, resources=resources)


@dataclass(frozen=True, slots=True)
class _BuiltinDatasetSpec:
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
    "interaction": _builtin("dronalize.datasets.interaction"),
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
    """Register one dataset descriptor in the in-memory registry."""
    if descriptor.name in _REGISTRY and _REGISTRY[descriptor.name] != descriptor:
        msg = f"Dataset '{descriptor.name}' is already registered."
        raise DatasetRegistryError(msg)
    _REGISTRY[descriptor.name] = descriptor


def get(name: str) -> DatasetSpec:
    """Return one registered or built-in dataset descriptor."""
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
    """Return the sorted list of available dataset names."""
    builtin_names = {
        name
        for name, spec in _builtin_datasets().items()
        if not _missing_optional_dependencies(spec)
    }
    return sorted(set(_REGISTRY) | builtin_names)


def _builtin_datasets() -> dict[str, _BuiltinDatasetSpec]:
    return _BUILTIN_DATASETS


@functools.cache
def _load_builtin_descriptor(name: str) -> DatasetSpec:
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

    descriptor = exported if spec.export_key is None else exported[spec.export_key]
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
    return tuple(
        module_name for module_name in spec.optional_dependencies if not _has_module(module_name)
    )


@functools.lru_cache
def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _missing_dependency_error(
    *, subject: str, spec: _BuiltinDatasetSpec, missing: tuple[str, ...]
) -> MissingOptionalDependencyError:
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
