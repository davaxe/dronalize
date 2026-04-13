"""Public runtime entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.datasets.registry import get
from dronalize.io.backends.registry import build_writer_factory
from dronalize.runtime._internal.resolve import build_job
from dronalize.runtime._internal.runner import open_job
from dronalize.runtime.types import ProcessResult

if TYPE_CHECKING:
    from dronalize.runtime.request import ProcessRequest
    from dronalize.runtime.types import RunPlan


def resolve_job(request: ProcessRequest) -> RunPlan:
    """Resolve one processing request into an execution-ready job."""
    descriptor = get(request.dataset)
    return build_job(descriptor=descriptor, request=request)


def process_dataset(request: ProcessRequest) -> ProcessResult:
    """Resolve and execute one dataset-processing request."""
    return run_job(resolve_job(request))


def run_job(job: RunPlan) -> ProcessResult:
    """Execute one resolved job and collect final counters and output paths."""
    with open_job(job) as run:
        run.executor.execute(writer_factory=build_writer_factory(job))
        job.write_manifests()
        progress = run.executor.progress()
        return ProcessResult(
            dataset=job.dataset,
            output_dir=job.output_dir,
            storage_backend=job.storage_backend,
            processed_sources=progress.processed_sources,
            processed_scenes=progress.processed_scenes,
            split_counts=progress.split_counts,
        )
