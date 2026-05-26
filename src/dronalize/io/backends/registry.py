"""Registry-driven writer backend resolution.

The runtime keeps backend selection separate from scene encoding. A resolved
`RunPlan` chooses a storage backend name, and this registry maps that name to a
builder that can create worker-local `DatasetWriter` instances.

The built-in registry entries are:

- `mds` for shard-based Mosaic Streaming output
- `pickle` for one pickled scene record per file
- `null` for dry-run style execution without persisted scene data
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from dronalize.core.errors import UnsupportedStorageBackendError
from dronalize.io.base import DatasetWriter
from dronalize.io.formats import StorageBackend, StorageBackendId, storage_backend_name

if TYPE_CHECKING:
    from dronalize.core.categories import DatasetSplit
    from dronalize.io.base import RecordTransform, SceneTransform
    from dronalize.runtime.types import ExecutionPlan

WriterFactory = Callable[[int | None], DatasetWriter]
WriterFactoryBuilder = Callable[["ExecutionPlan"], WriterFactory]
_WRITER_BACKENDS: dict[str, WriterFactoryBuilder] = {}
logger = logging.getLogger(__name__)


def registered_writer_backends() -> tuple[str, ...]:
    """Return registered writer backend names in deterministic order."""
    return tuple(sorted(_WRITER_BACKENDS))


def is_writer_backend_registered(backend: StorageBackendId) -> bool:
    """Return whether a writer backend has been registered."""
    return storage_backend_name(backend) in _WRITER_BACKENDS


def register_writer_backend(backend: StorageBackendId, builder: WriterFactoryBuilder) -> None:
    """Register a writer backend factory builder."""
    backend_name = storage_backend_name(backend)
    _WRITER_BACKENDS[backend_name] = builder
    logger.debug("Registered writer backend", extra={"storage_backend": backend_name})


def build_writer_factory(plan: ExecutionPlan) -> WriterFactory:
    """Build the writer factory for one resolved processing plan."""
    backend_name = storage_backend_name(plan.storage_backend)
    builder = _WRITER_BACKENDS.get(backend_name)
    if builder is None:
        raise UnsupportedStorageBackendError(backend_name, registered_writer_backends())
    logger.debug(
        "Building writer factory", extra={"dataset": plan.dataset, "storage_backend": backend_name}
    )
    return builder(plan)


def _build_mds_writer_factory(plan: ExecutionPlan) -> WriterFactory:
    from dronalize.io.backends.mds import MDSDatasetWriter  # noqa: PLC0415

    sample = plan.output_sample
    return MDSDatasetWriter.as_factory(
        plan.output_dir,
        config=plan.output,
        splits=_output_splits(plan),
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
    )


def _build_null_writer_factory(plan: ExecutionPlan) -> WriterFactory:
    from dronalize.io.backends.null import NullWriter  # noqa: PLC0415

    _ = plan
    return NullWriter.as_factory()


def _build_pickle_writer_factory(plan: ExecutionPlan) -> WriterFactory:
    from dronalize.io.backends.pickle import PickleWriter  # noqa: PLC0415

    sample = plan.output_sample
    return PickleWriter.as_factory(
        output_dir=plan.output_dir,
        config=plan.output,
        splits=_output_splits(plan),
        record_transform=(None if sample is None else sample.record_transform),
        scene_transform=(None if sample is None else sample.scene_transform),
    )


def _output_splits(plan: ExecutionPlan) -> tuple[DatasetSplit, ...] | None:
    return plan.assignment.output_splits(input_native_splits=plan.loader.read.native_splits)


register_writer_backend(StorageBackend.MDS, _build_mds_writer_factory)
register_writer_backend(StorageBackend.NULL, _build_null_writer_factory)
register_writer_backend(StorageBackend.PICKLE, _build_pickle_writer_factory)
