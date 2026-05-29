"""Public runtime entrypoints."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from dronalize.datasets.registry import get_dataset
from dronalize.io.backends.registry import build_writer_factory
from dronalize.io.base import storage_backend_name
from dronalize.runtime.executor import open_execution_session
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
        extra={
            "dataset": plan.dataset,
            "storage_backend": storage_backend_name(plan.storage_backend),
        },
    )
    start_time = time.time()
    with open_execution_session(plan) as run:
        execute_with_rich_progress(
            run.executor,
            lambda: run.executor.execute(writer_factory=build_writer_factory(plan)),
            enable=show_progress,
        )
        logger.debug("Execution complete, writing manifests", extra={"dataset": plan.dataset})
        plan.write_manifests()
        progress = run.executor.progress()
        logger.info(
            "Finished plan",
            extra={
                "dataset": plan.dataset,
                "processed_sources": progress.processed_sources,
                "selected_scenes": progress.selected_scenes,
            },
        )
        return ExecutionResult(
            dataset=plan.dataset,
            output_dir=plan.output_dir,
            storage_backend=plan.storage_backend,
            processed_sources=progress.processed_sources,
            candidate_scenes=progress.candidate_scenes,
            selected_scenes=progress.selected_scenes,
            split_counts={k: v for k, v in progress.split_counts.items() if isinstance(v, int)},
            elapsed_time_seconds=time.time() - start_time,
        )
