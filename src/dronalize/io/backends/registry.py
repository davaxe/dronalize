"""Registry-driven writer backend resolution.

The runtime keeps backend selection separate from scene encoding. A resolved
``RunPlan`` chooses a :class:`dronalize.io.formats.StorageBackend`, and this
registry maps that backend to a builder that can create worker-local
``DatasetWriter`` instances.

The built-in registry entries are:

- ``mds`` for shard-based Mosaic Streaming output
- ``pickle`` for one pickled scene record per file
- ``null`` for dry-run style execution without persisted scene data
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from dronalize.io.base import DatasetWriter
from dronalize.io.formats import StorageBackend

if TYPE_CHECKING:
    from dronalize.core.categories import DatasetSplit
    from dronalize.runtime.types import ExecutionPlan

WriterFactory = Callable[[int | None], DatasetWriter]
WriterFactoryBuilder = Callable[["ExecutionPlan"], WriterFactory]

_WRITER_BACKENDS: dict[StorageBackend, WriterFactoryBuilder] = {}


def register_writer_backend(backend: StorageBackend, builder: WriterFactoryBuilder) -> None:
    """Register a writer backend factory builder."""
    _WRITER_BACKENDS[backend] = builder


def build_writer_factory(job: ExecutionPlan) -> WriterFactory:
    """Build the writer factory for one resolved processing job."""
    builder = _WRITER_BACKENDS[job.storage_backend]
    return builder(job)


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
    split = plan.loader.split
    if split is None:
        return None
    return split.output_splits(available_native_splits=plan.descriptor.native_splits or None)


register_writer_backend(StorageBackend.MDS, _build_mds_writer_factory)
register_writer_backend(StorageBackend.NULL, _build_null_writer_factory)
register_writer_backend(StorageBackend.PICKLE, _build_pickle_writer_factory)
