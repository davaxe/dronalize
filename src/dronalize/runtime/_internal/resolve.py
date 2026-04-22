"""Runtime planning helpers for request-to-plan bootstrap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dronalize.core.errors as dronalize_exceptions
from dronalize.config.file import ProcessingConfig
from dronalize.config.models.split import (
    AssignConfig,
    NoAssign,
    PreserveNativeAssign,
    ReadConfig,
    ReadNative,
    SceneAssign,
    ShuffledTimeBlockAssign,
    SourceAssign,
    TimeBlockAssign,
)
from dronalize.config.reader import load_project_config
from dronalize.io.formats import StorageBackend
from dronalize.processing.models import AssignmentRequest
from dronalize.runtime.types import (
    ExecutionPlan,
    OutputPlan,
    compile_effective_scene_metrics,
    compile_loader_request,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.config.models import DatasetConfig
    from dronalize.config.runtime import RuntimeOverride
    from dronalize.datasets.registry import DatasetSpec
    from dronalize.runtime.types import ExecutionRequest


def build_plan(*, descriptor: DatasetSpec, request: ExecutionRequest) -> ExecutionPlan:
    """Build a full runtime plan from a public execution request."""
    _validate_input_path(request)
    include_map = request.include_map
    storage_backend = _resolve_storage_backend(request.storage_backend)
    resolved_config = _resolve_dataset_config(
        descriptor=descriptor, config_path=request.config_path, cli_override=request.overrides
    )
    loader_request = compile_loader_request(
        descriptor=descriptor, resolved_config=resolved_config, include_map=include_map
    )
    assignment_request = AssignmentRequest.from_config(resolved_config.assign, seed=request.seed)
    effective_history_frames, effective_future_frames, effective_sample_time = (
        compile_effective_scene_metrics(resolved_config)
    )
    if not _validate_read_support(descriptor, resolved_config.read):
        msg = f"Dataset {descriptor.name} does not support the requested read configuration."
        raise dronalize_exceptions.ConfigurationError(msg)
    if not _validate_assignment_support(descriptor, resolved_config.assign):
        msg = f"Dataset {descriptor.name} does not support the requested assignment configuration."
        raise dronalize_exceptions.ConfigurationError(msg)
    return ExecutionPlan(
        descriptor=descriptor,
        data_root=request.input_dir,
        output_dir=request.output_dir,
        storage_backend=storage_backend,
        runtime=resolved_config.runtime,
        output=OutputPlan(inner=resolved_config.output),
        loader=loader_request,
        assignment=assignment_request,
        map=loader_request.map,
        effective_history_frames=effective_history_frames,
        effective_future_frames=effective_future_frames,
        effective_sample_time=effective_sample_time,
        limit=request.limit,
        seed=request.seed,
        resolved_config=resolved_config,
    )


def _validate_read_support(spec: DatasetSpec, config: ReadConfig | None) -> bool:
    if config is None:
        return True
    match config.root:
        case ReadNative(splits=splits) if splits is not None:
            supported = spec.supported_native_splits
            if not supported:
                return False
            return set(splits).issubset(supported)
        case ReadNative(splits=None):
            return bool(spec.supported_native_splits)
        case _:
            return True


def _validate_assignment_support(spec: DatasetSpec, config: AssignConfig | None) -> bool:
    if config is None:
        return True
    support = spec.split_support
    match config.root:
        case NoAssign():
            return True
        case PreserveNativeAssign():
            return bool(spec.supported_native_splits)
        case TimeBlockAssign() | ShuffledTimeBlockAssign():
            return support.time_block
        case SceneAssign():
            return support.scene
        case SourceAssign():
            return support.source


def _validate_input_path(request: ExecutionRequest) -> None:
    if not request.input_dir.exists() and request.input_dir_exists:
        msg = f"Input directory {request.input_dir} does not exist."
        raise FileNotFoundError(msg)
    if request.input_dir_exists and not request.input_dir.is_dir():
        msg = f"Input directory {request.input_dir} is not a directory."
        raise NotADirectoryError(msg)


def _resolve_storage_backend(storage_backend: StorageBackend | str) -> StorageBackend:
    try:
        return StorageBackend(storage_backend)
    except ValueError as exc:
        raise dronalize_exceptions.UnsupportedStorageBackendError(
            storage_backend, tuple(f.value for f in StorageBackend)
        ) from exc


def _resolve_dataset_config(
    *, descriptor: DatasetSpec, config_path: Path | None, cli_override: RuntimeOverride
) -> DatasetConfig:

    project = _load_project_config(config_path)
    defaults = descriptor.default_config
    config = project.resolve(descriptor.name, defaults)
    return cli_override.apply_to(None).apply_to(config)


def _load_project_config(config_path: Path | None) -> ProcessingConfig:
    if config_path is None:
        return ProcessingConfig()
    return load_project_config(config_path)
