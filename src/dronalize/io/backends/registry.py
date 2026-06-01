"""Registry-driven writer backend resolution.

The runtime keeps backend selection separate from scene encoding. A resolved
`RunPlan` chooses a storage backend name, and this registry maps that name to a
builder that creates a writer provider for one execution run.

The built-in registry entries are:

- `mds` for shard-based Mosaic Streaming output
- `pickle` for one pickled scene record per file
- `null` for dry-run style execution without persisted scene data
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from dronalize.core.errors import UnsupportedStorageBackendError
from dronalize.io.base import (
    StorageBackend,
    StorageBackendId,
    WorkerWriterProvider,
    WriterProvider,
    storage_backend_name,
)

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.core.categories import DatasetSplit
    from dronalize.io.base import DatasetWriter, RecordTransform, SceneTransform
    from dronalize.runtime.types import ExecutionPlan, OutputPlan

WriterProviderBuilder = Callable[["ExecutionPlan"], WriterProvider]
_WRITER_BACKENDS: dict[str, WriterProviderBuilder] = {}
logger = logging.getLogger(__name__)


def registered_writer_backends() -> tuple[str, ...]:
    """Return registered writer backend names in deterministic order."""
    return tuple(sorted(_WRITER_BACKENDS))


def is_writer_backend_registered(backend: StorageBackendId) -> bool:
    """Return whether a writer backend has been registered."""
    return storage_backend_name(backend) in _WRITER_BACKENDS


def register_writer_backend(backend: StorageBackendId, builder: WriterProviderBuilder) -> None:
    """Register a writer backend provider builder."""
    backend_name = storage_backend_name(backend)
    _WRITER_BACKENDS[backend_name] = builder
    logger.debug("Registered writer backend", extra={"storage_backend": backend_name})


def build_writer_provider(plan: ExecutionPlan) -> WriterProvider:
    """Build the writer provider for one resolved processing plan."""
    backend_name = storage_backend_name(plan.storage_backend)
    builder = _WRITER_BACKENDS.get(backend_name)
    if builder is None:
        raise UnsupportedStorageBackendError(backend_name, registered_writer_backends())
    logger.debug(
        "Building writer provider", extra={"dataset": plan.dataset, "storage_backend": backend_name}
    )
    return builder(plan)


def _build_mds_writer_provider(plan: ExecutionPlan) -> WriterProvider:
    from dronalize.io.backends.mds import MDSDatasetWriter  # noqa: PLC0415

    sample = plan.output_sample
    splits = _output_splits(plan)
    return WorkerWriterProvider(
        create_worker=functools.partial(
            _create_mds_writer,
            output_dir=plan.output_dir,
            config=plan.output,
            splits=splits,
            parallel=plan.parallel,
            record_transform=(
                None
                if sample is None
                else cast("RecordTransform[dict[str, Any]] | None", sample.record_transform)
            ),
            scene_transform=(
                None
                if sample is None
                else cast("SceneTransform[dict[str, Any]] | None", sample.scene_transform)
            ),
            sample_columns=None if sample is None else sample.mds_columns,
        ),
        finalize=functools.partial(
            MDSDatasetWriter.finish_dataset,
            output_dir=plan.output_dir,
            splits=splits,
            parallel=plan.parallel,
        ),
    )


def _create_mds_writer(
    worker_id: int,
    *,
    output_dir: Path,
    config: OutputPlan,
    splits: tuple[DatasetSplit, ...] | None,
    parallel: bool,
    record_transform: RecordTransform[dict[str, Any]] | None,
    scene_transform: SceneTransform[dict[str, Any]] | None,
    sample_columns: dict[str, str] | None,
) -> DatasetWriter:
    from dronalize.io.backends.mds import MDSDatasetWriter  # noqa: PLC0415

    return MDSDatasetWriter(
        output_dir=output_dir,
        config=config,
        splits=splits,
        parallel=parallel,
        parallel_group=worker_id,
        record_transform=record_transform,
        scene_transform=scene_transform,
        sample_columns=sample_columns,
    )


def _build_null_writer_provider(plan: ExecutionPlan) -> WriterProvider:
    _ = plan
    return WorkerWriterProvider(create_worker=_create_null_writer)


def _create_null_writer(worker_id: int) -> DatasetWriter:
    from dronalize.io.backends.null import NullWriter  # noqa: PLC0415

    _ = worker_id
    return NullWriter()


def _build_pickle_writer_provider(plan: ExecutionPlan) -> WriterProvider:
    sample = plan.output_sample
    return WorkerWriterProvider(
        create_worker=functools.partial(
            _create_pickle_writer,
            output_dir=plan.output_dir,
            config=plan.output,
            splits=_output_splits(plan),
            record_transform=None if sample is None else sample.record_transform,
            scene_transform=None if sample is None else sample.scene_transform,
        )
    )


def _create_pickle_writer(
    worker_id: int,
    *,
    output_dir: Path,
    config: OutputPlan,
    splits: tuple[DatasetSplit, ...] | None,
    record_transform: RecordTransform[object] | None,
    scene_transform: SceneTransform[object] | None,
) -> DatasetWriter:
    from dronalize.io.backends.pickle import PickleWriter  # noqa: PLC0415

    return PickleWriter(
        output_dir=output_dir,
        identifier=worker_id,
        config=config,
        splits=splits,
        record_transform=record_transform,
        scene_transform=scene_transform,
    )


def _output_splits(plan: ExecutionPlan) -> tuple[DatasetSplit, ...] | None:
    return plan.assignment.output_splits(input_native_splits=plan.loader.read.native_splits)


register_writer_backend(StorageBackend.MDS, _build_mds_writer_provider)
register_writer_backend(StorageBackend.NULL, _build_null_writer_provider)
register_writer_backend(StorageBackend.PICKLE, _build_pickle_writer_provider)
