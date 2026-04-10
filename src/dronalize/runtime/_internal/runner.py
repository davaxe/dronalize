"""Internal runtime execution orchestration."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dronalize.runtime._internal.parallel import ParallelExecutor
from dronalize.runtime._internal.scene import SceneBuilder
from dronalize.runtime._internal.sequential import SequentialExecutor

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from dronalize.processing.loading.base import BaseSceneLoader, LoaderOptions
    from dronalize.processing.loading.loader import Source
    from dronalize.runtime._internal.executor import ObservableExecutor
    from dronalize.runtime.plans import RunPlan


@dataclass(slots=True)
class JobRun:
    job: RunPlan
    executor: ObservableExecutor


@contextmanager
def open_job(job: RunPlan) -> Generator[JobRun, None, None]:
    """Open one job with initialized resources, loader, builder, and executor."""
    with job.descriptor.open_resources(job.data_root, job.loader) as resources:
        loader = job.descriptor.build_loader(
            root=job.data_root, request=job.loader, resources=resources
        )
        sources = tuple(iter_sources(loader, job))
        builder = SceneBuilder.from_job(job)
        executor = _build_executor(job, loader, builder, sources)
        yield JobRun(job=job, executor=executor)


def iter_sources(
    loader: BaseSceneLoader[Any, LoaderOptions], job: RunPlan
) -> Iterable[Source[Any]]:
    """Yield runtime source objects for one loader and resolved job."""
    split = job.loader.split
    if split is not None and split.strategy == "native":
        read = split.read or job.descriptor.native_splits
        for native_split in read or ():
            for source in loader.sources_for_split(native_split):
                yield source.with_predefined_split(native_split)
        return
    if job.descriptor.native_splits:
        for native_split in job.descriptor.native_splits:
            for source in loader.sources_for_split(native_split):
                yield source.with_predefined_split(native_split)
        return
    yield from loader.discover_sources()


def _build_executor(
    job: RunPlan,
    loader: BaseSceneLoader[Any, LoaderOptions],
    builder: SceneBuilder,
    sources: tuple[Source[Any], ...],
) -> ObservableExecutor:
    if job.parallel:
        return ParallelExecutor(
            loader,
            builder,
            sources,
            workers=job.workers,
            chunksize=job.runtime.chunksize,
            limit=job.limit,
        )
    return SequentialExecutor(loader, builder, sources, limit=job.limit)
