"""Public runtime entrypoints."""

from __future__ import annotations

import logging
import shutil
import time
from typing import TYPE_CHECKING

from dronalize.datasets.registry import get_dataset
from dronalize.io.backends.provider import build_writer_provider
from dronalize.io.manifest import write_manifest
from dronalize.runtime.executor import open_executor
from dronalize.runtime.progress import execute_with_rich_progress
from dronalize.runtime.resolve import build_execution_plan
from dronalize.runtime.types import ExecutionResult

if TYPE_CHECKING:
    from dronalize.runtime.types import ExecutionPlan, ExecutionRequest


logger = logging.getLogger(__name__)


def resolve_request(request: ExecutionRequest) -> ExecutionPlan:
    """Resolve one processing request into an execution-ready plan.

    Parameters
    ----------
    request : ExecutionRequest
        The dataset-processing request to resolve. This includes all user-provided
        parameters, paths, and overrides.

    Returns
    -------
    ExecutionPlan
        A fully resolved execution plan, including all derived parameters,
        loader requests, and output plans. This plan is ready to be executed with
        [`execute_plan`][dronalize.runtime.execute_plan].

    """
    descriptor = get_dataset(request.dataset)
    logger.debug("Resolving execution request", extra={"dataset": request.dataset})
    return build_execution_plan(descriptor=descriptor, request=request)


def execute_request(request: ExecutionRequest, *, show_progress: bool = True) -> ExecutionResult:
    """Resolve and execute one dataset-processing request.

    This is a convenience method that combines the resolution and execution
    steps into a single call. It is suitable for simple use cases where the user
    does not need to inspect or modify the execution plan before running it.

    Parameters
    ----------
    request : ExecutionRequest
        The dataset-processing request to resolve and execute. This includes all
        user-provided parameters, paths, and overrides.
    show_progress : bool, optional
        Whether to display a rich progress bar during execution. Default is
        True.

    Returns
    -------
    ExecutionResult
        Summary of the execution, including counters, output paths, and elapsed
        time.
    """
    logger.info("Executing request", extra={"dataset": request.dataset})
    return execute_plan(resolve_request(request), show_progress=show_progress)


def execute_plan(plan: ExecutionPlan, *, show_progress: bool = True) -> ExecutionResult:
    """Execute one resolved plan and collect final counters and output paths.

    This will start the execution where the main objective is to process all
    scenes and write them to disk

    Parameters
    ----------
    plan : ExecutionPlan
        The execution plan to run. This should be a fully resolved plan.
    show_progress : bool, optional
        Whether to display a rich progress bar during execution. Default is
        True.

    Returns
    -------
    ExecutionResult
        Summary of the execution.
    """
    logger.info(
        "Executing plan",
        extra={"dataset": plan.dataset, "storage_backend": plan.storage_backend.value},
    )

    _prepare_output_directory(plan)
    start_time = time.perf_counter()
    with open_executor(plan) as executor:
        writer_provider = build_writer_provider(plan)
        progress = execute_with_rich_progress(
            executor,
            lambda: executor.execute(writer_provider),
            enable=show_progress,
        )
        logger.debug("Execution complete, writing manifests", extra={"dataset": plan.dataset})
        write_manifest(plan.output_dir, plan.manifest())
        logger.info(
            "Finished plan",
            extra={
                "dataset": plan.dataset,
                "processed_sources": progress.stats.processed_sources,
                "candidate_scenes": progress.stats.candidate_scenes,
                "written_scenes": progress.stats.written_scenes,
            },
        )

        return ExecutionResult(
            dataset=plan.dataset,
            output_dir=plan.output_dir,
            storage_backend=plan.storage_backend,
            stats=progress.stats,
            scene_limit=progress.scene_limit,
            cleanup_summary=executor.cleanup_summary(),
            elapsed_time_seconds=time.perf_counter() - start_time,
        )


def _prepare_output_directory(plan: ExecutionPlan) -> None:
    output_dir = plan.output_dir
    if not output_dir.exists():
        return

    try:
        _ = next(output_dir.iterdir())
    except StopIteration:
        return

    if not plan.overwrite:
        msg = (
            f"Output directory {output_dir} is not empty. "
            "Choose an empty directory or enable overwrite explicitly."
        )
        raise FileExistsError(msg)

    _ = shutil.rmtree(output_dir)
