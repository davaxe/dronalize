"""Multiprocessing executor for internal runtime exection."""

from __future__ import annotations

import functools
import multiprocessing as mp
from collections import deque
from multiprocessing.util import Finalize
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import override

from dronalize.core.scene import Scene
from dronalize.runtime._internal import state
from dronalize.runtime._internal.executor import Executor, WriterFactory
from dronalize.runtime._internal.state import Progress

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from multiprocessing.synchronize import Event

    from dronalize.core.typing import P
    from dronalize.io.base import DatasetWriter
    from dronalize.processing.loading.loader import Source
    from dronalize.runtime._internal.processor import RuntimeProcessor


ReturnT = TypeVar("ReturnT", int, list[Scene])
_ctx: state.WorkerRuntime


class ParallelExecutor(Executor):
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
    ) -> None:
        if workers is not None and workers <= 1:
            msg = "number of processes must be greater than 1 for parallel execution."
            raise ValueError(msg)
        self._processor: RuntimeProcessor = processor
        self._shared: state.SharedResources = state.SharedResources.create(scene_limit=limit)
        total_sources = processor.total_sources()
        self._chunksize: int = chunksize or self._optimal_chunksize(total_sources, workers)
        self._limit: int | None = limit
        self._processes: int | None = workers
        self._num_sources: int | None = total_sources
        self._screening_enabled: bool = processor.screening_enabled()
        self._running: bool = False

    @override
    def execute(self, writer_factory: WriterFactory) -> None:
        _ = deque(
            self._execute_parallel(
                self._process_fn_write,
                self._processor.iter_sources(),
                _init_write_worker,
                *(self._shared, self._processor, writer_factory),
            ),
            maxlen=0,
        )

    @override
    def execute_yield(self) -> Iterator[Scene]:
        for scenes in self._execute_parallel(
            self._process_fn_yield,
            self._processor.iter_sources(),
            _init_worker,
            self._shared,
            self._processor,
        ):
            yield from scenes

    @override
    def progress(self) -> Progress:
        with self._shared.progress.snapshot_lock:
            return Progress(
                running=self._running,
                processed_sources=self._shared.progress.source_counter.value,
                candidate_scenes=self._shared.progress.candidate_scene_counter.value,
                selected_scenes=self._shared.progress.selected_scene_counter.value,
                active_workers=self._shared.progress.active_workers.value,
                total_sources=self._num_sources,
                scene_limit=self._limit,
                split_counts=self._shared.progress.split_counts(),
                screening_enabled=self._screening_enabled,
            )

    @override
    def progress_event(self) -> Event:
        return self._shared.progress.update_event

    @override
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def _process_fn_write(source: Source[Any]) -> int:
        if _ctx.writer is None:
            msg = "DatasetWriter was not initialized for this worker process."
            raise ValueError(msg)
        if _ctx.processor is None:
            msg = "Runtime processor was not initialized for this worker process."
            raise ValueError(msg)
        if _ctx.shared.progress.selected_scene_limit_reached(_ctx.shared.scene_limit):
            return 0
        selected_scenes = 0
        for scene in ParallelExecutor._generate_scenes(_ctx.processor, source):
            _ctx.shared.progress.record_split(scene.split_assignment)
            _ctx.writer.write(scene)
            selected_scenes += 1
        _ = _ctx.shared.progress.increment_source()
        return selected_scenes

    @staticmethod
    def _process_fn_yield(source: Source[Any]) -> list[Scene]:
        if _ctx.processor is None:
            msg = "Runtime processor was not initialized for this worker process."
            raise ValueError(msg)
        if _ctx.shared.progress.selected_scene_limit_reached(_ctx.shared.scene_limit):
            return []
        _ = _ctx.shared.progress.increment_source()
        scenes = list(ParallelExecutor._generate_scenes(_ctx.processor, source))
        for scene in scenes:
            _ctx.shared.progress.record_split(scene.split_assignment)
        return scenes

    @staticmethod
    def _generate_scenes(processor: RuntimeProcessor, source: Source[Any]) -> Iterator[Scene]:
        if _ctx.shared.progress.selected_scene_limit_reached(_ctx.shared.scene_limit):
            return
        claim_selected_scene = functools.partial(
            _ctx.shared.progress.claim_selected_scene, _ctx.shared.scene_limit
        )
        for candidate in processor.iter_candidates(source):
            _ = _ctx.shared.progress.record_candidate_scene()
            if not candidate.passes_screening:
                continue
            scene_number = claim_selected_scene()
            if scene_number is None:
                return
            yield processor.materialize(candidate, scene_number)

    def _execute_parallel(
        self,
        process_fn: Callable[[Source[Any]], ReturnT],
        payloads: Iterable[Source[Any]],
        initializer: Callable[P, object],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterable[ReturnT]:
        self._shared.reset()
        pool_initializer = functools.partial(initializer, *args, **kwargs)
        self._running = True
        self.progress_event().set()
        with mp.Pool(self._processes, initializer=pool_initializer) as pool:
            yield from pool.imap_unordered(process_fn, payloads, self._chunksize)
        self._running = False
        self.progress_event().set()

    @staticmethod
    def _optimal_chunksize(num_sources: int | None, num_processes: int | None) -> int:
        if num_sources is None:
            return 1
        process_count = num_processes or mp.cpu_count()
        chunksize, extra = divmod(num_sources, process_count * 4)
        chunksize += int(extra > 0)
        return max(chunksize, 1)


def _init_worker(
    shared: state.SharedResources,
    processor: RuntimeProcessor | None = None,
    *,
    with_finalize: bool = True,
) -> None:
    global _ctx  # noqa: PLW0603
    worker_id = shared.registry.next_worker()
    shared.progress.worker_started()
    _ctx = state.WorkerRuntime(shared=shared, worker_id=worker_id, processor=processor)
    if with_finalize:

        def cleanup() -> None:
            _ = _ctx.shared.progress.worker_stopped()

        _ = Finalize(obj=None, callback=cleanup, exitpriority=10)


def _init_write_worker(
    shared: state.SharedResources,
    processor: RuntimeProcessor,
    writer_factory: Callable[[int], DatasetWriter],
) -> None:
    global _ctx  # noqa: PLW0602
    _init_worker(shared, processor, with_finalize=False)
    writer: DatasetWriter | None = None
    try:
        writer = writer_factory(_ctx.worker_id)
        _ctx.writer = writer
    except Exception:
        _ = _ctx.shared.progress.worker_stopped()
        raise

    def cleanup() -> None:
        current_writer = _ctx.writer
        if current_writer is None:
            return
        try:
            with _ctx.progress_snapshot_lock():
                active_before = _ctx.active_workers()
                current_writer.finish_local()
                if active_before == 1:
                    current_writer.finish_final()
        finally:
            _ = _ctx.shared.progress.worker_stopped()

    _ = Finalize(obj=None, callback=cleanup, exitpriority=10)
