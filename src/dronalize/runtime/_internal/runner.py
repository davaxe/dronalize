"""Internal runtime execution orchestration."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dronalize.runtime._internal.parallel import ParallelExecutor
from dronalize.runtime._internal.processor import RuntimeProcessor
from dronalize.runtime._internal.sequential import SequentialExecutor

if TYPE_CHECKING:
    from collections.abc import Generator

    from dronalize.runtime._internal.executor import Executor
    from dronalize.runtime.types import ExecutionPlan


@dataclass(slots=True)
class ExecutionSession:
    plan: ExecutionPlan
    executor: Executor


@contextmanager
def open_execution_session(plan: ExecutionPlan) -> Generator[ExecutionSession, None, None]:
    """Open one plan with initialized resources, processor, and executor."""
    with plan.descriptor.open_resources(plan.data_root, plan.loader) as resources:
        loader = plan.descriptor.build_loader(
            root=plan.data_root, request=plan.loader, resources=resources
        )
        processor = RuntimeProcessor.from_plan(plan, loader)
        executor = _build_executor(plan, processor)
        yield ExecutionSession(plan=plan, executor=executor)


def _build_executor(plan: ExecutionPlan, processor: RuntimeProcessor) -> Executor:
    if plan.parallel:
        return ParallelExecutor(
            processor, workers=plan.workers, chunksize=plan.runtime.chunksize, limit=plan.limit
        )
    return SequentialExecutor(processor, limit=plan.limit)
