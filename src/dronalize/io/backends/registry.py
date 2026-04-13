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
    from dronalize.runtime.types import RunPlan

WriterFactory = Callable[[int | None], DatasetWriter]
WriterFactoryBuilder = Callable[["RunPlan"], WriterFactory]

_WRITER_BACKENDS: dict[StorageBackend, WriterFactoryBuilder] = {}


def register_writer_backend(backend: StorageBackend, builder: WriterFactoryBuilder) -> None:
    """Register a writer backend factory builder."""
    _WRITER_BACKENDS[backend] = builder


def build_writer_factory(job: RunPlan) -> WriterFactory:
    """Build the writer factory for one resolved processing job."""
    builder = _WRITER_BACKENDS[job.storage_backend]
    return builder(job)


def _build_mds_writer_factory(job: RunPlan) -> WriterFactory:
    from dronalize.io.backends.mds import MDSDatasetWriter  # noqa: PLC0415

    return MDSDatasetWriter.as_factory(
        job.output_dir, config=job.output, splits=_output_splits(job), parallel=job.parallel
    )


def _build_null_writer_factory(job: RunPlan) -> WriterFactory:
    from dronalize.io.backends.null import NullWriter  # noqa: PLC0415

    _ = job
    return NullWriter.as_factory(log=True)


def _build_pickle_writer_factory(job: RunPlan) -> WriterFactory:
    from dronalize.io.backends.pickle import PickleWriter  # noqa: PLC0415

    return PickleWriter.as_factory(
        output_dir=job.output_dir, config=job.output, splits=_output_splits(job)
    )


def _output_splits(job: RunPlan) -> tuple[DatasetSplit, ...] | None:
    split = job.loader.split
    if split is None:
        return None
    return split.output_splits(available_native_splits=job.descriptor.native_splits or None)


register_writer_backend(StorageBackend.MDS, _build_mds_writer_factory)
register_writer_backend(StorageBackend.NULL, _build_null_writer_factory)
register_writer_backend(StorageBackend.PICKLE, _build_pickle_writer_factory)
