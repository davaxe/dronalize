"""Runtime planning helpers for request-to-plan bootstrap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dronalize.core.errors as dronalize_exceptions
from dronalize.config.project import ProjectConfig
from dronalize.config.reader import load_project_config
from dronalize.io.formats import StorageBackend
from dronalize.runtime.plans import (
    OutputPlan,
    RunPlan,
    compile_effective_scene_metrics,
    compile_loader_request,
)
from dronalize.runtime.request import PlanningRequest

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.config.sections import DatasetConfig
    from dronalize.datasets.registry import DatasetSpec
    from dronalize.runtime.request import ProcessRequest


def build_job(*, descriptor: DatasetSpec, request: ProcessRequest) -> RunPlan:
    """Build a full runtime plan from a public process request."""
    _validate_input_path(request)
    include_map = request.include_map
    storage_backend = _resolve_storage_backend(request.storage_backend)
    planning = PlanningRequest(
        descriptor=descriptor,
        config_path=request.config_path,
        overrides=request.overrides,
        seed=request.seed,
        include_map=include_map,
    )
    resolved_config = _resolve_dataset_config(descriptor=descriptor, planning=planning)
    loader_request = compile_loader_request(
        descriptor=descriptor,
        resolved_config=resolved_config,
        seed=request.seed,
        include_map=include_map,
    )
    effective_history_frames, effective_future_frames, effective_sample_time = (
        compile_effective_scene_metrics(resolved_config)
    )
    return RunPlan(
        descriptor=descriptor,
        planning=planning,
        data_root=request.input_dir,
        output_dir=request.output_dir,
        storage_backend=storage_backend,
        runtime=resolved_config.runtime,
        output=OutputPlan(inner=resolved_config.output),
        loader=loader_request,
        map=loader_request.map,
        effective_history_frames=effective_history_frames,
        effective_future_frames=effective_future_frames,
        effective_sample_time=effective_sample_time,
        limit=request.limit,
        seed=request.seed,
        resolved_config=resolved_config,
    )


def _validate_input_path(request: ProcessRequest) -> None:
    if not request.input_dir.exists() and request.input_dir_exists:
        msg = f"Input directory {request.input_dir} does not exist."
        raise FileNotFoundError(msg)


def _resolve_storage_backend(storage_backend: StorageBackend | str) -> StorageBackend:
    try:
        return StorageBackend(storage_backend)
    except ValueError as exc:
        raise dronalize_exceptions.UnsupportedStorageBackendError(
            storage_backend, tuple(f.value for f in StorageBackend)
        ) from exc


def _resolve_dataset_config(*, descriptor: DatasetSpec, planning: PlanningRequest) -> DatasetConfig:

    cli_override = planning.overrides
    project = _load_project_config(planning.config_path)
    defaults = descriptor.default_config
    config = project.resolve(planning.descriptor.name, defaults)
    return cli_override.apply_to(None).apply_to(config)


def _load_project_config(config_path: Path | None) -> ProjectConfig:
    if config_path is None:
        return ProjectConfig()
    return load_project_config(config_path)
