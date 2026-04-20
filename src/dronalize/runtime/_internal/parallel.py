"""Multiprocessing executor for internal runtime exection."""

from __future__ import annotations

import functools
import multiprocessing as mp
from collections import deque
from multiprocessing.util import Finalize
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from typing_extensions import override

from dronalize.core.scene import Scene
from dronalize.core.typing import SourceT
from dronalize.runtime._internal import state
from dronalize.runtime._internal.executor import ObservableExecutor, WriterFactory
from dronalize.runtime._internal.state import Progress

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from multiprocessing.synchronize import Event

    from dronalize.core.typing import P
    from dronalize.io.base import DatasetWriter
    from dronalize.processing.loading.base import BaseSceneLoader, DatasetOptionsModel
    from dronalize.processing.loading.loader import Source
    from dronalize.runtime._internal.scene import SceneBuilder


ReturnT = TypeVar("ReturnT", int, list[Scene])
_ctx: state.WorkerRuntime


class ParallelExecutor(ObservableExecutor, Generic[SourceT]):
    def __init__(
        self,
        loader: BaseSceneLoader[SourceT, Any],
        builder: SceneBuilder,
        sources: Iterable[Source[SourceT]],
        *,
        chunksize: int | None = None,
        workers: int | None = None,
        limit: int | None = None,
    ) -> None:
        if workers is not None and workers <= 1:
            msg = "number of processes must be greater than 1 for parallel execution."
            raise ValueError(msg)
        self._loader: BaseSceneLoader[SourceT, DatasetOptionsModel] = loader
        self._builder: SceneBuilder = builder
        self._sources: Iterable[Source[SourceT]] = sources
        self._shared: state.SharedResources = state.SharedResources.create(scene_limit=limit)
        self._chunksize: int = chunksize or self._optimal_chunksize(loader.num_sources(), workers)
        self._limit: int | None = limit
        self._processes: int | None = workers
        self._num_sources: int | None = loader.num_sources()
        self._running: bool = False

    @override
    def execute(
        self, writer_factory: WriterFactory, finalize: Callable[[DatasetWriter], None] | None = None
    ) -> None:
        _ = deque(
            self._execute_parallel(
                self._process_fn_write,
                self._sources,
                _init_write_worker,
                *(self._shared, self._loader, self._builder, writer_factory, finalize),
            ),
            maxlen=0,
        )

    @override
    def execute_yield(self) -> Iterator[Scene]:
        for scenes in self._execute_parallel(
            self._process_fn_yield, self._sources, _init_worker, self._shared
        ):
            yield from scenes

    @override
    def progress(self) -> Progress:
        with self._shared.progress.snapshot_lock:
            return Progress(
                running=self._running,
                processed_sources=self._shared.progress.source_counter.value,
                processed_scenes=self._shared.progress.scene_counter.value,
                active_workers=self._shared.progress.active_workers.value,
                total_sources=self._num_sources,
                total_scenes=self._limit,
                split_counts=self._shared.progress.split_counts(),
            )

    @override
    def progress_event(self) -> Event:
        return self._shared.progress.update_event

    @override
    def is_running(self) -> bool:
        return self._running

    @property
    def split_counts(self) -> dict[str, int]:
        with self._shared.progress.snapshot_lock:
            return self._shared.progress.split_counts()

    @staticmethod
    def _process_fn_write(source: Source[Any]) -> int:
        if _ctx.writer is None:
            msg = "DatasetWriter was not initialized for this worker process."
            raise ValueError(msg)
        if _ctx.loader is None or _ctx.builder is None:
            msg = "Loader runtime was not initialized for this worker process."
            raise ValueError(msg)
        _ = _ctx.shared.progress.increment_source()
        processed_scenes = 0
        for scene in ParallelExecutor._generate_scenes(_ctx.loader, _ctx.builder, source):
            _ctx.shared.progress.record_split(scene.split_assignment)
            _ctx.writer.write(scene)
            processed_scenes += 1
        return processed_scenes

    @staticmethod
    def _process_fn_yield(source: Source[Any]) -> list[Scene]:
        if _ctx.loader is None or _ctx.builder is None:
            msg = "Loader runtime was not initialized for this worker process."
            raise ValueError(msg)
        _ = _ctx.shared.progress.increment_source()
        scenes = list(ParallelExecutor._generate_scenes(_ctx.loader, _ctx.builder, source))
        for scene in scenes:
            _ctx.shared.progress.record_split(scene.split_assignment)
        return scenes

    @staticmethod
    def _generate_scenes(
        loader: BaseSceneLoader[Any, Any], builder: SceneBuilder, source: Source[Any]
    ) -> Iterator[Scene]:
        for processed in builder.prepare_source(loader, source):
            scene_number = _ctx.shared.progress.claim_scene(_ctx.shared.scene_limit)
            if scene_number is None:
                return
            yield builder.create_scene(loader, processed, source, scene_number)

    def _execute_parallel(
        self,
        process_fn: Callable[[Source[SourceT]], ReturnT],
        payloads: Iterable[Source[SourceT]],
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
            pool.close()
            pool.join()
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


def _init_worker(shared: state.SharedResources, *, with_finalize: bool = True) -> None:
    global _ctx  # noqa: PLW0603
    worker_id = shared.registry.next_worker()
    shared.progress.worker_started()
    _ctx = state.WorkerRuntime(shared=shared, worker_id=worker_id)
    if with_finalize:

        def cleanup() -> None:
            assert _ctx is not None
            _ = _ctx.shared.progress.worker_stopped()

        _ = Finalize(obj=None, callback=cleanup, exitpriority=10)


def _init_write_worker(
    shared: state.SharedResources,
    loader: BaseSceneLoader[Any, DatasetOptionsModel],
    builder: SceneBuilder,
    writer_factory: Callable[[int], DatasetWriter],
    finalize: Callable[[DatasetWriter], None] | None,
) -> None:
    global _ctx  # noqa: PLW0602
    _init_worker(shared, with_finalize=False)
    assert _ctx is not None
    _ctx.loader = loader
    _ctx.builder = builder
    writer: DatasetWriter | None = None
    try:
        writer = writer_factory(_ctx.worker_id)
        _ctx.writer = writer
    except Exception:
        _ = _ctx.shared.progress.worker_stopped()
        raise

    def cleanup() -> None:
        assert _ctx is not None
        current_writer = _ctx.writer
        assert current_writer is not None
        try:
            with _ctx.progress_snapshot_lock():
                active_before = _ctx.active_workers()
                if finalize is not None:
                    finalize(current_writer)
                else:
                    current_writer.finish_local()
                if active_before == 1:
                    current_writer.finish_final()
        finally:
            _ = _ctx.shared.progress.worker_stopped()

    _ = Finalize(obj=None, callback=cleanup, exitpriority=10)
