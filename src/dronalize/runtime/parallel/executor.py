"""Multiprocessing executor for parallel scene generation and writing."""

from __future__ import annotations

import functools
import multiprocessing as mp
from collections import deque
from multiprocessing.util import Finalize
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar

from typing_extensions import Self, override

from dronalize._internal.typing import P, SourceT
from dronalize.core.scene import Scene
from dronalize.runtime.executor import ObservableExecutor, Progress, WriterFactory
from dronalize.runtime.parallel import _state

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from multiprocessing.synchronize import Event

    from dronalize.io.backends.base import DatasetWriter
    from dronalize.processing.loading.base import BaseSceneLoader
    from dronalize.processing.loading.loader import Source


ReturnT = TypeVar("ReturnT", int, list[Scene])
# Type var for static methods (they cannot inherit from a generic type)
_S = TypeVar("_S")

_ctx: _state.WorkerRuntime  # Global state for each worker process


class _SceneProcessArgs(NamedTuple, Generic[SourceT]):
    """Arguments for processing a single source into scenes."""

    source: Source[SourceT]
    loader: BaseSceneLoader[SourceT, Any]


class ParallelExecutor(ObservableExecutor, Generic[SourceT]):
    """Parallel writing executor backed by a processable loader.

    This executor uses Python's `multiprocessing` module to distribute source
    processing across multiple CPU cores. The number of processes and the task
    chunk size can be configured depending on workload.

    Notes
    -----
    Some practical considerations when using this executor:

    1. If underlying sources are very small and quick to process, the overhead
       of multiprocessing can outweigh the benefits. Increasing `chunksize` may
       help in such cases.

    2. Since wrapped loaders may themselves use multi-threaded libraries such as
       Polars, resource contention can occur between multiprocessing and
       internal threading. In practice this may require reducing the number of
       worker processes or setting environment variables such as
       `POLARS_MAX_THREADS`.

    3. The wrapped loader is sent to each worker process, so it should be
       lightweight and picklable. Avoid eager loading of large resources in the
       loader constructor when possible.
    """

    def __init__(
        self,
        inner: BaseSceneLoader[SourceT, Any],
        *,
        chunksize: int | None = None,
        workers: int | None = None,
        limit: int | None = None,
    ) -> None:
        """Initialize the executor.

        Parameters
        ----------
        inner : DatasetLoader[SourceT]
            Loader responsible for turning sources into scene data.
        chunksize : int, optional
            Number of sources sent to each worker task batch. If omitted, an
            automatic value is chosen from the number of sources and processes
            when possible.
        workers : int, optional
            Number of worker processes. If omitted, the default multiprocessing
            process count is used.
        limit : int, optional
            An optional limit on the total number of scenes to process. If
            `None`, all scenes produced by the loader are processed.
        """
        if workers is not None and workers <= 1:
            msg = "number of processes must be greater than 1 for ParallelExecutor."
            raise ValueError(msg)

        self._inner: BaseSceneLoader[SourceT, Any] = inner
        self._shared: _state.SharedResources = _state.SharedResources.create(scene_limit=limit)
        self._chunksize: int = chunksize or self._optimal_chunksize(inner.num_sources(), workers)
        self._limit: int | None = limit
        self._processes: int | None = workers
        self._num_sources: int | None = inner.num_sources()
        self._running: bool = False

    def workers(self, workers: int | None) -> Self:
        """Set the number of worker processes.

        Parameters
        ----------
        workers : int or None
            Number of worker processes. If None, the default multiprocessing
            process count is used.

        Returns
        -------
        Self
            The executor instance.
        """
        if workers is not None and workers <= 1:
            msg = "number of processes must be greater than 1 for ParallelExecutor."
            raise ValueError(msg)

        self._processes = workers
        return self

    def chunksize(self, chunksize: int) -> Self:
        """Set the task chunk size used by the multiprocessing pool.

        Larger chunk sizes can reduce scheduling overhead, but may also reduce
        load balancing quality when source processing times vary a lot.

        Parameters
        ----------
        chunksize : int
            Number of sources per pool task batch.

        Returns
        -------
        Self
            The executor instance.
        """
        if chunksize <= 0:
            msg = "chunksize must be a positive integer."
            raise ValueError(msg)

        self._chunksize = chunksize
        return self

    @override
    def execute(
        self, writer_factory: WriterFactory, finalize: Callable[[DatasetWriter], None] | None = None
    ) -> None:
        """Process scenes in parallel and write them inside worker processes.

        This mode initializes one `DatasetWriter` per worker process. Each worker
        writes scenes locally without sending them back to the parent process.

        A unique integer worker ID is passed to `writer_factory`. This can be
        used to create one output shard per worker, for example.

        Parameters
        ----------
        writer_factory : Callable[[int], DatasetWriter]
            Factory called once per worker process with that worker's integer
            ID.
        finalize : Callable[[DatasetWriter], None], optional
            Optional custom per-worker finalization hook. If omitted,
            `writer.finish_local()` is called instead.

        """
        payloads = (_SceneProcessArgs(source, self._inner) for source in self._inner.sources())
        _ = deque(
            self._execute_parallel(
                self._process_fn_write,
                payloads,
                _init_write_worker,
                *(self._shared, writer_factory, finalize),
            ),
            maxlen=0,
        )

    @override
    def progress(self) -> Progress:
        """Return a snapshot of the current parallel execution progress.

        This method is intended to be called from the main process while the
        executor is running.

        Returns
        -------
        Progress
            Current progress information.
        """
        with self._shared.progress.snapshot_lock:
            return Progress(
                running=self._running,
                processed_sources=self._shared.progress.source_counter.value,
                processed_scenes=self._shared.progress.scene_counter.value,
                active_workers=self._shared.progress.active_workers.value,
                total_sources=self._num_sources,
                total_scenes=self._limit,
            )

    @override
    def progress_event(self) -> Event:
        """Return the shared event used to signal progress changes."""
        return self._shared.progress.update_event

    @override
    def is_running(self) -> bool:
        """Return whether the parallel executor is actively processing."""
        return self._running

    @staticmethod
    def _process_fn_write(args: _SceneProcessArgs[_S]) -> int:
        """Worker function that processes one source and writes scenes locally.

        Returns
        -------
        int
            Number of scenes written from the source.
        """
        if _ctx.writer is None:
            msg = "DatasetWriter was not initialized for this worker process."
            raise ValueError(msg)

        processed_scenes: int = 0
        for scene in ParallelExecutor._generate_scenes(args.loader, args.source):
            _ = _ctx.writer.write(scene)
            processed_scenes += 1

        _ = _ctx.shared.progress.increment_source()
        return processed_scenes

    @staticmethod
    def _generate_scenes(loader: BaseSceneLoader[_S, Any], source: Source[_S]) -> Iterator[Scene]:
        """Process a single source and yield numbered scenes.

        Scene numbers are assigned from a shared global counter when scenes are
        created inside worker processes.
        """
        for processed in loader.process_next(source):
            scene_number = _ctx.shared.progress.claim_scene(_ctx.shared.scene_limit)
            if scene_number is None:
                return
            yield loader.create_scene(processed, source, scene_number)

    def _execute_parallel(
        self,
        process_fn: Callable[[_SceneProcessArgs[SourceT]], ReturnT],
        payloads: Iterable[_SceneProcessArgs[SourceT]],
        initializer: Callable[P, object],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterable[ReturnT]:
        """Run the pool and yield task results.

        This is the central multiprocessing loop used by `execute()`. It
        resets shared counters before creating the pool.

        Parameters
        ----------
        process_fn : Callable[[PayloadT], ReturnT]
            Worker task function.
        payloads : Iterable[PayloadT]
            Task payload stream.
        initializer : Callable
            Worker initializer called once per process.
        *args : Any
            Positional initializer arguments.
        **kwargs : Any
            Keyword initializer arguments.

        Yields
        ------
        ReturnT
            Task results yielded from the pool iterator.
        """
        self._shared.reset()
        pool_initializer = functools.partial(initializer, *args, **kwargs)

        self._running = True
        self.progress_event().set()
        with mp.Pool(self._processes, initializer=pool_initializer) as pool:
            yield from pool.imap_unordered(process_fn, payloads, self._chunksize)
            pool.close()
            pool.join()

        self._running = False
        # Notify eventual progress listeners that processing has completed and
        # the final state can be observed
        self.progress_event().set()

    @staticmethod
    def _optimal_chunksize(num_sources: int | None, num_processes: int | None) -> int:
        """Compute a simple default chunk size.

        The heuristic targets roughly four chunks per process, with a minimum of
        one.
        """
        if num_sources is None:
            return 1
        process_count = num_processes or mp.cpu_count()
        chunksize, extra = divmod(num_sources, process_count * 4)
        chunksize += int(extra > 0)
        return max(chunksize, 1)


def _init_worker(shared: _state.SharedResources, *, with_finalize: bool = True) -> None:
    """Initialize process-local worker runtime.

    Parameters
    ----------
    shared : SharedResources
        Shared multiprocessing state.
    with_finalize : bool, optional
        If True, register a finalizer that decrements the active worker count.
        This is disabled for write workers because they use a custom finalizer
        that must also handle writer cleanup.
    """
    global _ctx  # noqa: PLW0603

    worker_id = shared.registry.next_worker()
    shared.progress.worker_started()
    _ctx = _state.WorkerRuntime(shared=shared, worker_id=worker_id)

    if with_finalize:

        def cleanup() -> None:
            assert _ctx is not None
            _ = _ctx.shared.progress.worker_stopped()

        _ = Finalize(obj=None, callback=cleanup, exitpriority=10)


def _init_write_worker(
    shared: _state.SharedResources,
    writer_factory: Callable[[int], DatasetWriter],
    finalize: Callable[[DatasetWriter], None] | None,
) -> None:
    """Initialize a worker for `execute()` write mode.

    This creates a worker-local writer and installs a custom finalizer that
    performs local writer cleanup, optional custom finalization, and exactly one
    final global finish step.

    """
    global _ctx  # noqa: PLW0602
    _init_worker(shared, with_finalize=False)

    assert _ctx is not None
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
