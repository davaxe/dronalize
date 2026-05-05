"""Registry-driven writer backend resolution.

The runtime keeps backend selection separate from scene encoding. A resolved
`RunPlan` chooses a :class:`dronalize.io.formats.StorageBackend`, and this
registry maps that backend to a builder that can create worker-local
`DatasetWriter` instances.

The built-in registry entries are:

- `mds` for shard-based Mosaic Streaming output
- `pickle` for one pickled scene record per file
- `null` for dry-run style execution without persisted scene data
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from dronalize.core.errors import UnsupportedStorageBackendError
from dronalize.io.base import DatasetWriter
from dronalize.io.formats import StorageBackend

if TYPE_CHECKING:
    from dronalize.core.categories import DatasetSplit
    from dronalize.runtime.types import ExecutionPlan

WriterFactory = Callable[[int | None], DatasetWriter]
WriterFactoryBuilder = Callable[["ExecutionPlan"], WriterFactory]

_WRITER_BACKENDS: dict[StorageBackend, WriterFactoryBuilder] = {}
logger = logging.getLogger(__name__)


def register_writer_backend(backend: StorageBackend, builder: WriterFactoryBuilder) -> None:
    """Register a writer backend factory builder."""
    _WRITER_BACKENDS[backend] = builder
    logger.debug("Registered writer backend", extra={"storage_backend": backend.value})


def build_writer_factory(plan: ExecutionPlan) -> WriterFactory:
    """Build the writer factory for one resolved processing plan."""
    builder = _WRITER_BACKENDS.get(plan.storage_backend)
    if builder is None:
        raise UnsupportedStorageBackendError(
            plan.storage_backend.value, tuple(backend.value for backend in _WRITER_BACKENDS)
        )
    logger.debug(
        "Building writer factory",
        extra={"dataset": plan.dataset, "storage_backend": plan.storage_backend.value},
    )
    return builder(plan)


def _build_mds_writer_factory(plan: ExecutionPlan) -> WriterFactory:
    from dronalize.io.backends.mds import MDSDatasetWriter  # noqa: PLC0415

    return MDSDatasetWriter.as_factory(
        plan.output_dir, config=plan.output, splits=_output_splits(plan), parallel=plan.parallel
    )


def _build_null_writer_factory(plan: ExecutionPlan) -> WriterFactory:
    from dronalize.io.backends.null import NullWriter  # noqa: PLC0415

    _ = plan
    return NullWriter.as_factory(log=True)


def _build_pickle_writer_factory(plan: ExecutionPlan) -> WriterFactory:
    from dronalize.io.backends.pickle import PickleWriter  # noqa: PLC0415

    return PickleWriter.as_factory(
        output_dir=plan.output_dir, config=plan.output, splits=_output_splits(plan)
    )


def _output_splits(plan: ExecutionPlan) -> tuple[DatasetSplit, ...] | None:
    return plan.assignment.output_splits(input_native_splits=plan.loader.read.native_splits)


register_writer_backend(StorageBackend.MDS, _build_mds_writer_factory)
register_writer_backend(StorageBackend.NULL, _build_null_writer_factory)
register_writer_backend(StorageBackend.PICKLE, _build_pickle_writer_factory)
