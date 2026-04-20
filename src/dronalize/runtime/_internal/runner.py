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

    from dronalize.processing.loading.base import BaseSceneLoader, DatasetOptionsModel
    from dronalize.processing.loading.loader import Source
    from dronalize.runtime._internal.executor import ObservableExecutor
    from dronalize.runtime.types import ExecutionPlan


@dataclass(slots=True)
class ExecutionSession:
    plan: ExecutionPlan
    executor: ObservableExecutor


@contextmanager
def open_execution_session(plan: ExecutionPlan) -> Generator[ExecutionSession, None, None]:
    """Open one plan with initialized resources, loader, builder, and executor."""
    with plan.descriptor.open_resources(plan.data_root, plan.loader) as resources:
        loader = plan.descriptor.build_loader(
            root=plan.data_root, request=plan.loader, resources=resources
        )
        sources = iter_sources(loader, plan)
        builder = SceneBuilder.from_plan(plan)
        executor = _build_executor(plan, loader, builder, sources)
        yield ExecutionSession(plan=plan, executor=executor)


def iter_sources(
    loader: BaseSceneLoader[Any, DatasetOptionsModel], plan: ExecutionPlan
) -> Iterable[Source[Any]]:
    """Yield runtime source objects for one loader and resolved plan."""
    split = plan.loader.split
    if split is not None and split.strategy == "native":
        read = split.read or plan.descriptor.native_splits
        for native_split in read or ():
            for source in loader.sources_for_split(native_split):
                yield source.with_predefined_split(native_split)
        return
    if plan.descriptor.native_splits:
        for native_split in plan.descriptor.native_splits:
            for source in loader.sources_for_split(native_split):
                yield source.with_predefined_split(native_split)
        return
    yield from loader.discover_sources()


def _build_executor(
    plan: ExecutionPlan,
    loader: BaseSceneLoader[Any, DatasetOptionsModel],
    builder: SceneBuilder,
    sources: Iterable[Source[Any]],
) -> ObservableExecutor:
    if plan.parallel:
        return ParallelExecutor(
            loader,
            builder,
            sources,
            workers=plan.workers,
            chunksize=plan.runtime.chunksize,
            limit=plan.limit,
        )
    return SequentialExecutor(loader, builder, sources, limit=plan.limit)
