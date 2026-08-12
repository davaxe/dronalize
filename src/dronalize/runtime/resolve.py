"""Runtime planning helpers for request-to-plan bootstrap."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import dronalize.core.errors as dronalize_exceptions
from dronalize.config.models import (
    AssignConfig,
    DatasetConfig,
    NoAssign,
    PreserveNativeAssign,
    ReadConfig,
    ReadNative,
    SceneAssign,
    ShuffledTimeBlockAssign,
    SourceAssign,
    TimeBlockAssign,
)
from dronalize.config.parse import ProjectConfig, parse_config
from dronalize.io.base import StorageBackend
from dronalize.processing.models import SplitAssignmentPlan
from dronalize.runtime.types import ExecutionPlan, build_loader_plan, resolve_effective_scene_window

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.config.models import RuntimeOverride
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
    resolved_config, selected_task = _resolve_dataset_config(
        descriptor=descriptor, config_path=request.config_path, cli_override=request.overrides
    )
    _validate_read_support(descriptor, resolved_config.read)
    _validate_assignment_support(descriptor, resolved_config.assign)
    _validate_feature_support(descriptor, resolved_config)
    _validate_temporal_support(descriptor, resolved_config)
    loader_request = build_loader_plan(
        descriptor=descriptor, resolved_config=resolved_config, include_map=include_map
    )
    assignment_request = SplitAssignmentPlan.from_config(resolved_config.assign, seed=request.seed)
    effective_horizon_frames, effective_prediction_bounds, effective_sample_time = (
        resolve_effective_scene_window(resolved_config)
    )
    logger.debug(
        "Built execution plan",
        extra={
            "dataset": descriptor.name,
            "storage_backend": storage_backend,
            "parallel": resolved_config.runtime.jobs > 1,
            "include_map": loader_request.map is not None,
        },
    )
    return ExecutionPlan(
        descriptor=descriptor,
        selected_task=selected_task,
        data_root=request.input_dir,
        output_dir=request.output_dir,
        storage_backend=StorageBackend(storage_backend),
        runtime=resolved_config.runtime,
        loader=loader_request,
        assignment=assignment_request,
        map=loader_request.map,
        effective_horizon_frames=effective_horizon_frames,
        effective_prediction_bounds=effective_prediction_bounds,
        effective_sample_time=effective_sample_time,
        output_transform=request.output_transform,
        limit=request.limit,
        seed=request.seed,
        overwrite=request.overwrite,
        resolved_config=resolved_config,
    )


def _validate_read_support(descriptor: DatasetDescriptor, config: ReadConfig | None) -> None:
    if not isinstance(config, ReadNative):
        return
    supported = descriptor.supported_native_splits
    if supported and (config.splits is None or set(config.splits).issubset(supported)):
        return
    msg = f"Dataset {descriptor.name} does not support the requested read configuration."
    raise dronalize_exceptions.ConfigurationError(msg)


def _validate_assignment_support(
    descriptor: DatasetDescriptor, config: AssignConfig | None
) -> None:
    if config is None:
        return
    support = descriptor.split_support
    match config:
        case NoAssign():
            supported = True
        case PreserveNativeAssign():
            supported = bool(descriptor.supported_native_splits)
        case TimeBlockAssign() | ShuffledTimeBlockAssign():
            supported = support.time_block
        case SceneAssign():
            supported = support.scene
        case SourceAssign():
            supported = support.source
    if not supported:
        msg = f"Dataset {descriptor.name} does not support the requested assignment configuration."
        raise dronalize_exceptions.ConfigurationError(msg)


def _validate_feature_support(descriptor: DatasetDescriptor, config: DatasetConfig) -> None:
    if config.scenes.lane_change is None:
        return
    if config.scenes.window is None:
        msg = "Lane-change sampling requires window sampling to be enabled."
        raise dronalize_exceptions.ConfigurationError(msg)
    if not descriptor.feature_support.lane_change_sampling:
        msg = f"Dataset {descriptor.name} does not support lane-change sampling."
        raise dronalize_exceptions.ConfigurationError(msg)


def _validate_temporal_support(descriptor: DatasetDescriptor, config: DatasetConfig) -> None:
    support = descriptor.temporal_support
    window = config.scenes.window
    if support is None or window is None:
        return

    windowing = support.windowing
    if window.policy not in windowing.supported_policies:
        msg = (
            f"Dataset {descriptor.name} does not support window policy '{window.policy}'. "
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
        f"Dataset {descriptor.name} supports windows up to {max_frames} source frames, "
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
    input_path = request.input_dir.resolve()
    output_path = request.output_dir.resolve()
    if (
        input_path == output_path
        or input_path.is_relative_to(output_path)
        or output_path.is_relative_to(input_path)
    ):
        msg = "Input and output directories must not overlap."
        raise dronalize_exceptions.ConfigurationError(msg)


def _resolve_storage_backend(storage_backend: StorageBackend | str) -> StorageBackend:
    try:
        resolved: StorageBackend = StorageBackend(storage_backend)
    except ValueError as exc:
        raise dronalize_exceptions.UnsupportedStorageBackendError(
            storage_backend, tuple(sb.value for sb in StorageBackend)
        ) from exc
    return resolved


def _resolve_dataset_config(
    *, descriptor: DatasetDescriptor, config_path: Path | None, cli_override: RuntimeOverride
) -> tuple[DatasetConfig, str | None]:
    project = parse_config(config_path) if config_path else ProjectConfig()
    defaults = descriptor.default_config
    selected_task = descriptor.default_task

    config = project.resolve_dataset_config(
        descriptor.name,
        defaults,
        named_tasks=descriptor.tasks,
        default_task=descriptor.default_task,
    )
    project_task = project.task_selection_for(descriptor.name)
    if isinstance(project_task, str):
        selected_task = project_task
    elif project_task is not None:
        selected_task = None

    config = cli_override.merge_into(config)
    logger.debug("Resolved dataset config", extra={"dataset": descriptor.name})
    return config, selected_task
