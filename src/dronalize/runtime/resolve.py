"""Runtime planning helpers for request-to-plan bootstrap."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import dronalize.core.errors as dronalize_exceptions
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
from dronalize.config.parse import parse_config
from dronalize.config.project import ProjectConfig
from dronalize.io.backends.registry import is_writer_backend_registered, registered_writer_backends
from dronalize.io.formats import StorageBackend, storage_backend_name
from dronalize.processing.models import SplitAssignmentPlan
from dronalize.runtime.types import (
    ExecutionPlan,
    OutputPlan,
    build_loader_plan,
    resolve_effective_scene_window,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.config.models import DatasetConfig
    from dronalize.config.runtime import RuntimeOverride
    from dronalize.datasets.registry import DatasetDescriptor
    from dronalize.runtime.types import ExecutionRequest


logger = logging.getLogger(__name__)


def build_execution_plan(
    *, descriptor: DatasetDescriptor, request: ExecutionRequest
) -> ExecutionPlan:
    """Build a full runtime plan from a public execution request."""
    _validate_input_path(request)
    _validate_output_path(request)
    include_map = request.include_map
    storage_backend = _resolve_storage_backend(request.storage_backend)
    resolved_config = _resolve_dataset_config(
        descriptor=descriptor, config_path=request.config_path, cli_override=request.overrides
    )
    if not _validate_read_support(descriptor, resolved_config.read):
        msg = f"Dataset {descriptor.name} does not support the requested read configuration."
        raise dronalize_exceptions.ConfigurationError(msg)
    if not _validate_assignment_support(descriptor, resolved_config.assign):
        msg = f"Dataset {descriptor.name} does not support the requested assignment configuration."
        raise dronalize_exceptions.ConfigurationError(msg)
    if not _validate_feature_support(descriptor, resolved_config):
        msg = f"Dataset {descriptor.name} does not support lane-change sampling."
        raise dronalize_exceptions.ConfigurationError(msg)
    _validate_temporal_support(descriptor, resolved_config)
    loader_request = build_loader_plan(
        descriptor=descriptor, resolved_config=resolved_config, include_map=include_map
    )
    assignment_request = SplitAssignmentPlan.from_config(resolved_config.assign, seed=request.seed)
    effective_horizon_frames, effective_default_observation_length, effective_sample_time = (
        resolve_effective_scene_window(resolved_config)
    )
    logger.debug(
        "Built execution plan",
        extra={
            "dataset": descriptor.name,
            "storage_backend": storage_backend_name(storage_backend),
            "parallel": resolved_config.runtime.jobs > 1,
            "include_map": loader_request.map is not None,
        },
    )
    return ExecutionPlan(
        descriptor=descriptor,
        data_root=request.input_dir,
        output_dir=request.output_dir,
        storage_backend=storage_backend,
        runtime=resolved_config.runtime,
        output=OutputPlan(config=resolved_config.output),
        loader=loader_request,
        assignment=assignment_request,
        map=loader_request.map,
        effective_horizon_frames=effective_horizon_frames,
        effective_default_observation_length=effective_default_observation_length,
        effective_sample_time=effective_sample_time,
        limit=request.limit,
        seed=request.seed,
        resolved_config=resolved_config,
    )


def _validate_read_support(spec: DatasetDescriptor, config: ReadConfig | None) -> bool:
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


def _validate_assignment_support(spec: DatasetDescriptor, config: AssignConfig | None) -> bool:
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


def _validate_feature_support(spec: DatasetDescriptor, config: DatasetConfig) -> bool:
    if config.scenes.lane_change is None:
        return True
    if config.scenes.window is None:
        msg = "Lane-change sampling requires window sampling to be enabled."
        raise dronalize_exceptions.ConfigurationError(msg)
    return spec.feature_support.lane_change_sampling


def _validate_temporal_support(spec: DatasetDescriptor, config: DatasetConfig) -> None:
    support = spec.temporal_support
    window = config.scenes.window
    if support is None or window is None:
        return

    windowing = support.windowing
    if window.policy not in windowing.supported_policies:
        msg = (
            f"Dataset {spec.name} does not support window policy '{window.policy}'. "
            f"Supported policies: {', '.join(windowing.supported_policies)}."
        )
        raise dronalize_exceptions.ConfigurationError(msg)
    if windowing.validation == "off":
        return

    requested_frames = config.scenes.horizon_frames
    max_frames = (
        windowing.max_window_frames
        if windowing.max_window_frames is not None
        else support.source_frame_bounds.max_frames
    )
    if max_frames is None or requested_frames <= max_frames:
        return

    msg = (
        f"Dataset {spec.name} supports windows up to {max_frames} source frames, "
        f"but the resolved scene window requests {requested_frames} frames."
    )
    if windowing.validation == "warn":
        logger.warning(msg)
        return
    raise dronalize_exceptions.ConfigurationError(msg)


def _validate_input_path(request: ExecutionRequest) -> None:
    if not request.input_dir.exists() and request.input_dir_exists:
        msg = f"Input directory {request.input_dir} does not exist."
        raise FileNotFoundError(msg)
    if request.input_dir_exists and not request.input_dir.is_dir():
        msg = f"Input directory {request.input_dir} is not a directory."
        raise NotADirectoryError(msg)


def _validate_output_path(request: ExecutionRequest) -> None:
    if request.output_dir.exists() and not request.output_dir.is_dir():
        msg = f"Output directory {request.output_dir} is not a directory."
        raise NotADirectoryError(msg)


def _resolve_storage_backend(storage_backend: StorageBackend | str) -> StorageBackend | str:
    try:
        resolved: StorageBackend | str = StorageBackend(storage_backend)
    except ValueError:
        resolved = str(storage_backend)
    if is_writer_backend_registered(resolved):
        return resolved
    raise dronalize_exceptions.UnsupportedStorageBackendError(
        storage_backend_name(resolved), registered_writer_backends()
    )


def _resolve_dataset_config(
    *, descriptor: DatasetDescriptor, config_path: Path | None, cli_override: RuntimeOverride
) -> DatasetConfig:
    project = parse_config(config_path) if config_path is not None else ProjectConfig()
    defaults = descriptor.default_config
    config = project.resolve_dataset_config(descriptor.name, defaults)
    logger.debug("Resolved dataset config", extra={"dataset": descriptor.name})
    return cli_override.merge_into(None).merge_into(config)
