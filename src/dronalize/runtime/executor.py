"""Internal runtime executors."""

from __future__ import annotations

import functools
import logging
import multiprocessing as mp
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from multiprocessing.util import Finalize
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from typing_extensions import override

from dronalize.runtime.accounting import (
    CleanupSummaryAccumulator,
    LocalRunAccounting,
    SharedRunAccounting,
    iter_scenes_from_source,
)
from dronalize.runtime.processor import RuntimeProcessor
from dronalize.runtime.state import Progress, SharedResources

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator
    from multiprocessing.context import BaseContext
    from multiprocessing.pool import Pool

    from dronalize.core.scene import Scene
    from dronalize.core.typing import P
    from dronalize.io.base import DatasetWriter, WriterProvider
    from dronalize.processing.loading.models import DatasetSource
    from dronalize.runtime.types import CleanupSummary, ExecutionPlan

AnyEvent = Event | threading.Event
ReturnT = TypeVar("ReturnT")
_ctx: WorkerRuntime

logger = logging.getLogger(__name__)


class ProgressSource(ABC):
    """Read-only progress interface for execution observers."""

    @abstractmethod
    def snapshot(self) -> Progress:
        """Return a point-in-time progress snapshot."""
        ...

    @abstractmethod
    def changed(self) -> AnyEvent:
        """Return the event set whenever progress changes."""
        ...

    def wait_for_change(self) -> Progress:
        """Wait for the next progress change and return progress."""
        _ = self.changed().wait()
        self.changed().clear()
        return self.snapshot()


class Executor(Protocol):
    """Shared protocol for sequential and parallel execution of a plan."""

    @property
    def progress(self) -> ProgressSource:
        """Return the progress interface for this executor."""
        ...

    def execute(self, writer_provider: WriterProvider) -> Progress:
        """Execute the plan and return the final progress snapshot.

        The writer provider owns creation of worker-local writers and any
        dataset-wide finalization required after workers finish.

        """
        ...

    def cleanup_summary(self) -> CleanupSummary | None:
        """Return aggregated cleanup statistics collected during execution."""
        ...


@dataclass(slots=True)
class ExecutionSession:
    plan: ExecutionPlan
    executor: Executor


@dataclass(slots=True)
class WorkerRuntime:
    """Runtime objects available inside one worker process."""

    shared: SharedResources
    worker_id: int
    processor: RuntimeProcessor | None = None
    writer: DatasetWriter | None = None


@contextmanager
def open_execution_session(plan: ExecutionPlan) -> Generator[ExecutionSession]:
    """Open one plan with its map provider, processor, and executor."""
    with plan.descriptor.open_resources(plan.data_root, plan.loader) as map_provider:
        logger.debug("Opening execution session", extra={"dataset": plan.dataset})
        loader = plan.descriptor.build_loader(
            root=plan.data_root, request=plan.loader, map_provider=map_provider
        )
        processor = RuntimeProcessor.from_plan(plan, loader)
        executor = _build_executor(plan, processor)
        yield ExecutionSession(plan=plan, executor=executor)


def _build_executor(plan: ExecutionPlan, processor: RuntimeProcessor) -> Executor:
    if plan.parallel:
        logger.debug("Using parallel executor", extra={"dataset": plan.dataset})
        return ParallelExecutor(
            processor, workers=plan.workers, chunksize=plan.runtime.chunksize, limit=plan.limit
        )
    logger.debug("Using sequential executor", extra={"dataset": plan.dataset})
    return SequentialExecutor(processor, limit=plan.limit)


class SequentialExecutor(Executor, ProgressSource):
    """Single-process executor for internal runtime execution.

    Parameters
    ----------
    processor: RuntimeProcessor
        The runtime processor to execute, containing logic and configurations.
    limit: int | None, optional
        An optional limit on the total number of scenes to select. If None, no
        limit will be applied. Default is None.
    """

    def __init__(self, processor: RuntimeProcessor, *, limit: int | None = None) -> None:
        self._processor: RuntimeProcessor = processor
        self._screening_enabled: bool = processor.screening_enabled()
        self._total_sources: int | None = processor.total_sources()
        self._update_event: threading.Event = threading.Event()
        self._running: bool = False
        self._accounting: LocalRunAccounting = LocalRunAccounting(
            limit=limit, update_event=self._update_event
        )

    @property
    @override
    def progress(self) -> ProgressSource:
        return self

    @override
    def execute(self, writer_provider: WriterProvider) -> Progress:
        writer = writer_provider.open_worker(0)
        try:
            for scene in self._iter_scenes():
                writer.write(scene)
                self._accounting.record_written(scene.split_assignment)
        finally:
            try:
                writer.finish_local()
            finally:
                writer_provider.finish_final()

        return self.snapshot()

    @override
    def snapshot(self) -> Progress:
        return self._accounting.snapshot(
            running=self._running,
            total_sources=self._total_sources,
            active_workers=1 if self._running else 0,
            screening_enabled=self._screening_enabled,
        )

    @override
    def cleanup_summary(self) -> CleanupSummary | None:
        return self._accounting.cleanup_summary()

    @override
    def changed(self) -> threading.Event:
        return self._update_event

    def _iter_scenes(self) -> Iterator[Scene]:
        self._running = True
        self._update_event.set()

        try:
            for source in self._processor.iter_sources():
                if self._accounting.limit_reached():
                    break
                yield from iter_scenes_from_source(self._processor, source, self._accounting)
        finally:
            self._running = False
            self._update_event.set()


class ParallelExecutor(Executor, ProgressSource):
    """Parallel executor for internal runtime execution.

    Parameters
    ----------
    processor: RuntimeProcessor
        The runtime processor to execute, containing logic and cofigurations.
    chunksize: int | None, optional
        The number of sources to process in each worker batch. If None, an
        optimal chunksize will be estimated based on a simple heuristic. Default
        is None.
    workers: int | None, optional
        The number of worker processes to use for parallel execution. If None,
        the number of CPU cores will be used. Default is None.
    limit: int | None, optional
        An optional limit on the total number of scenes to select across all
        workers. If None, no limit will be applied. Default is None.
    """

    def __init__(
        self,
        processor: RuntimeProcessor,
        *,
        chunksize: int | None = None,
        workers: int | None = None,
        limit: int | None = None,
        mp_context: BaseContext | None = None,
    ) -> None:
        if workers is not None and workers <= 1:
            msg = "number of processes must be greater than 1 for parallel execution."
            raise ValueError(msg)
        self._processor: RuntimeProcessor = processor
        total_sources = processor.total_sources()
        self._chunksize: int = chunksize or self._optimal_chunksize(total_sources, workers)
        self._limit: int | None = limit
        self._processes: int | None = workers
        self._num_sources: int | None = total_sources
        self._screening_enabled: bool = processor.screening_enabled()
        self._running: bool = False
        self._cleanup_accumulator: CleanupSummaryAccumulator = CleanupSummaryAccumulator()
        self._mp_context: BaseContext = mp_context or mp.get_context("spawn")
        self._shared: SharedResources = SharedResources.create(
            scene_limit=limit, mp_context=self._mp_context
        )

    @property
    @override
    def progress(self) -> ProgressSource:
        return self

    @override
    def execute(self, writer_provider: WriterProvider) -> Progress:
        for cleanup_summary in self._execute_parallel(
            self._process_fn_write,
            self._processor.iter_sources(),
            _init_write_worker,
            self._shared,
            self._processor,
            writer_provider,
        ):
            self._cleanup_accumulator.merge(cleanup_summary)
        writer_provider.finish_final()
        return self.snapshot()

    @override
    def snapshot(self) -> Progress:
        return self._shared.progress.snapshot(
            running=self._running,
            total_sources=self._num_sources,
            scene_limit=self._limit,
            screening_enabled=self._screening_enabled,
        )

    @override
    def changed(self) -> Event:
        return self._shared.progress.update_event

    @override
    def cleanup_summary(self) -> CleanupSummary | None:
        return self._cleanup_accumulator.freeze()

    @staticmethod
    def _process_fn_write(source: DatasetSource[Any]) -> CleanupSummary | None:
        if _ctx.writer is None:
            msg = "DatasetWriter was not initialized for this worker process."
            raise ValueError(msg)
        if _ctx.processor is None:
            msg = "Runtime processor was not initialized for this worker process."
            raise ValueError(msg)

        accounting = SharedRunAccounting(
            progress=_ctx.shared.progress, limit=_ctx.shared.scene_limit
        )
        if accounting.limit_reached():
            return None

        for scene in iter_scenes_from_source(_ctx.processor, source, accounting):
            _ctx.writer.write(scene)
            accounting.record_written(scene.split_assignment)

        return accounting.cleanup_summary()

    def _execute_parallel(
        self,
        process_fn: Callable[[DatasetSource[Any]], ReturnT],
        payloads: Iterable[DatasetSource[Any]],
        initializer: Callable[P, object],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterable[ReturnT]:
        self._shared.reset()
        pool_initializer = functools.partial(initializer, *args, **kwargs)
        self._running = True
        self.changed().set()
        pool: Pool | None = None
        completed = False
        try:
            pool = self._mp_context.Pool(self._processes, initializer=pool_initializer)
            yield from pool.imap_unordered(process_fn, payloads, self._chunksize)
            completed = True
        finally:
            if pool is not None:
                if completed:
                    pool.close()
                else:
                    pool.terminate()
                pool.join()
            self._running = False
            self.changed().set()

    @staticmethod
    def _optimal_chunksize(num_sources: int | None, num_processes: int | None) -> int:
        if num_sources is None:
            return 1
        process_count = num_processes or mp.cpu_count()
        chunksize, extra = divmod(num_sources, process_count * 4)
        chunksize += int(extra > 0)
        return max(chunksize, 1)


def _init_worker(
    shared: SharedResources,
    processor: RuntimeProcessor | None = None,
    *,
    with_finalize: bool = True,
) -> None:
    global _ctx  # noqa: PLW0603
    worker_id = shared.registry.next_worker()
    shared.progress.worker_started()
    _ctx = WorkerRuntime(shared=shared, worker_id=worker_id, processor=processor)
    if with_finalize:

        def cleanup() -> None:
            _ctx.shared.progress.worker_stopped()

        _ = Finalize(obj=None, callback=cleanup, exitpriority=10)


def _init_write_worker(
    shared: SharedResources, processor: RuntimeProcessor, writer_provider: WriterProvider
) -> None:
    global _ctx  # noqa: PLW0602
    _init_worker(shared, processor, with_finalize=False)
    writer: DatasetWriter | None = None
    try:
        writer = writer_provider.open_worker(_ctx.worker_id)
        _ctx.writer = writer
    except Exception:
        _ctx.shared.progress.worker_stopped()
        raise

    def cleanup() -> None:
        current_writer = _ctx.writer
        if current_writer is None:
            return
        try:
            current_writer.finish_local()
        finally:
            _ctx.shared.progress.worker_stopped()

    _ = Finalize(obj=None, callback=cleanup, exitpriority=10)
