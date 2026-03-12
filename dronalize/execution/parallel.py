from __future__ import annotations

import contextlib
import functools
import multiprocessing as mp
import threading
from collections import deque
from dataclasses import dataclass
from multiprocessing.util import Finalize
from typing import TYPE_CHECKING, Concatenate, Generic, NamedTuple, NoReturn, TypeVar

import tqdm
from typing_extensions import Self, TypedDict, override

from dronalize._internal._types import P, PayloadT, SourceT
from dronalize.execution.common import ProgressBar
from dronalize.loading import ProcessableLoader, SceneLoader, Source
from dronalize.scene import Scene

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator
    from multiprocessing.sharedctypes import Synchronized
    from multiprocessing.synchronize import Lock

    from dronalize.categories import DatasetSplit
    from dronalize.loading import SceneWriter


ReturnT = TypeVar("ReturnT", int, list[Scene])
# Type var for static methods (they can't inherit from a generic type)
_S = TypeVar("_S")


@dataclass(frozen=True, slots=True)
class WorkerInfo:
    """Information about the current worker process."""

    worker_id: int
    """Id of the current worker process, starting from 0."""
    num_workers: int
    """Total number of worker processes."""
    remaining_workers: int
    """Remaining number of worker processes."""


class _TqdmArgs(TypedDict):
    total: int | None
    disable: bool
    unit: str
    desc: str
    colour: str
    unit_scale: bool | float


class ParallelExecutor(SceneLoader, Generic[SourceT]):
    """Parallel scene processor that wraps a processable loader.

    It uses Python's multiprocessing module to parallelize the processing of
    sources across multiple CPU cores. The number of processes and chunksize can
    be configured depending on workload.

    By default the order of resulting scenes when using the `scenes` method will
    not be deterministic across runs or systems due to the use of
    `imap_unordered` for better performance. If a deterministic order is
    required, the `maintain_order` flag can be set to True, which will use
    `imap` instead of `imap_unordered`.

    Practical Considerations
    ------------------------
    Using multiprocessing can significantly speed up the processing of large
    datasets, however there are some important considerations to keep in mind:

    1. If the underlying sources are small and quick to process, the overhead of
    multiprocessing may outweigh the benefits. One option is to increase the
    `chunksize` to reduce overhead in this case.

    2. If sources are instead large, for example `WaymoLoader` and its tfrecord
    files, then other issues may arise such as increased memory usage and
    possibly issues with inter-process communication. In these cases it is a
    good idea to avoid the `scenes` method, and instead use the
    `scenes_callback` method with a callback function that performs side effects
    (e.g., saving to file). If an error related to "Too many opened files" is
    encountered when using `scenes`, this is a strong indication that the
    underlying sources are too large for `scenes` and `scenes_callback` should
    be used.

    3. If the scenes include a lot of data (e.g. large DataFrames or maps)
    sending them between processes can be costly, and `scenes_callback` should
    be preferred when possible. In extreme cases using multiprocessing can slow
    down processing due to the overhead.

    4. Since the wrapped loaders use Polars, which also uses multiple cores for
    processing, there may be some contention for CPU resources between the
    multiprocessing in `ParallelExecutor` and the multithreading in Polars.
    Setting the environment variable `POLARS_MAX_THREADS=X`, where X is a number
    that balances the workload between `ParallelExecutor` and Polars. An
    alternative option is to lower the `processes` parameter in this class.

    5. The input dataloader will be sent to each worker process. This means that
    it should be as lightweight as possible, and possibly avoid eager loading of
    data in its constructor. It also needs to be picklable, in order to be sent
    to worker processes.

    """

    def __init__(
        self,
        inner: ProcessableLoader[SourceT],
        *,
        chunksize: int | None = None,
        processes: int | None = None,
        maintain_order: bool = False,
        progress_bar: ProgressBar | bool = ProgressBar.NONE,
    ) -> None:
        """Initialize with the given loader and multiprocessing configuration.

        Progress bar behaviour:

        - If `progress_bar` is set to `ProgressBar.SOURCES`, a progress bar
          will be shown that tracks the number of sources processed.
        - If `progress_bar` is set to `ProgressBar.SCENES`, a progress bar
          will be shown that tracks the number of scenes processed.

        No matter what option is chosen, the other quantity (sources or scenes)
        will be tracked in the progress bar's postfix for visibility. Using
        `ProgressBar.SOURCES` is generally recommended since the total number
        of sources is often known beforehand, while the number of scenes is
        usually not known beforehand (this results in only showing the number of
        processed scenes without a total, which can be less useful).

        Parameters
        ----------
        inner : BaseSceneLoader[SourceT]
            The underlying loader to use for processing each source.
        chunksize : int, optional
            The number of sources to process in each batch when using
            multiprocessing. Larger chunksizes can reduce overhead but may lead
            to less even load distribution across processes. If `None`, tries
            to automatically determine an optimal chunksize based on the number
            of sources and processes (when possible).
        processes : int, optional
            The number of processes to use for multiprocessing. If None, uses
            the number of CPU cores.
        maintain_order : bool, optional
            If True, the order of resulting scenes from `scenes` will be
            maintained using `imap` instead of `imap_unordered`.
        progress_bar : ProgressBar or bool, optional
            Whether to show a progress bar using tqdm. `True` is equivalent to
            `ProgressBar.SOURCES`. See behaviour description above.

        """
        if processes is not None and processes <= 1:
            msg = "number of processes must be greater than 1 for ParallelExecutor."
            raise ValueError(msg)

        self._inner: ProcessableLoader[SourceT] = inner
        self._mp_worker_counter: Synchronized[int] = mp.Value("i", 0)
        self._mp_scene_counter: Synchronized[int] = mp.Value("i", 0)
        self._mp_source_counter: Synchronized[int] = mp.Value("i", 0)
        self._mp_split_queue: mp.Queue[DatasetSplit | None] | None = None
        self._mp_info_lock: Lock = mp.Lock()
        self._chunksize: int = chunksize or self._optimal_chunksize(inner.num_sources(), processes)
        self._processes: int | None = processes
        self._maintain_order: bool = maintain_order
        self._progress_bar: ProgressBar = (
            ProgressBar(int(progress_bar)) if isinstance(progress_bar, bool) else progress_bar
        )

    def processes(self, processes: int | None) -> Self:
        """Set the number of processes to use for multiprocessing.

        Parameters
        ----------
        processes : int or None
            Number of worker processes. If None, uses the number of CPU cores.

        Returns
        -------
        Self
            The loader instance with the updated process count.

        """
        self._processes = processes
        return self

    def chunksize(self, chunksize: int) -> Self:
        """Set chunksize used when creating the `multiprocessing.Pool`.

        Larger chunksizes can reduce overhead but may lead to less even load
        distribution across processes.

        Parameters
        ----------
        chunksize : int
            Number of sources per batch sent to each worker process.

        Returns
        -------
        Self
            The loader instance with the updated chunksize.

        """
        self._chunksize = chunksize
        return self

    @override
    def scenes(self) -> Iterable[Scene]:
        """Process scenes in parallel and yield them one by one.

        This uses `multiprocessing` to process data in parallel. It works by
        creating a pool of worker processes that process batches (chunks,
        determined by `chunksize`) of sources. The results are collected and
        yielded as scenes.

        Note that each source might yield multiple scenes (for example if
        windowing is used), and each worker will process all scenes before
        returning to the main process.

        Yields
        ------
        Scene
            Processed scenes one at a time.

        """
        payloads = (_ProcessArgs(s, self._inner) for s in self._inner.sources())
        for scenes in self._execute_parallel(
            self._process_fn, payloads, _init_worker, *self._init_args
        ):
            yield from scenes

    @override
    def scenes_callback(
        self,
        callback: Callable[Concatenate[Scene, P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Process scenes using a callback function instead of yielding them.

        This method allows you to provide a callback function that will be
        called for each processed scene. This can be more efficient than
        yielding scenes if you want to perform side effects (e.g., saving to
        disk) without needing to store all scenes in memory at once.

        The callback should be defined as a top-level function (not a lambda or
        nested function) to ensure it is picklable and can be sent to worker
        processes.

        Parameters
        ----------
        callback : Callable
            A callable that takes a Scene as input and performs side effects
            (e.g., saving to disk). This function will be called for each
            processed scene.
        *args : Any
            Additional positional arguments to pass to the callback function.
        **kwargs : Any
            Additional keyword arguments to pass to the callback function.

        """
        # Bind args and kwargs early to avoid passing them through _ProcessArgs
        bound_callback = functools.partial(callback, *args, **kwargs)
        payloads = (_ProcessArgs(s, self._inner, bound_callback) for s in self._inner.sources())
        _ = deque(
            self._execute_parallel(
                self._process_fn_callback, payloads, _init_worker, *self._init_args
            ),
            maxlen=0,
        )

    @override
    def write_scenes(
        self,
        writer_factory: Callable[[int], SceneWriter],
        finalize: Callable[[SceneWriter], None] | None = None,
        split_generator: Iterable[DatasetSplit] | None = None,
    ) -> None:
        """Process scenes in parallel and write them using a SceneWriter.

        This is a customized method that initialize one SceneWriter per worker
        process using the provided `writer_factory`, and then calls the `write`
        method of the SceneWriter for each processed scene. This is useful for
        efficiently writing large datasets in parallel without needing to send
        scenes back to the main process. A unique integer (0, 1, ..., n) is
        passed to the `writer_factory` for each worker process, which can be used
        to create separate output files for each process if desired.

        Parameters
        ----------
        writer_factory : Callable[[int, P], SceneWriter]
            A factory function that takes a worker ID and additional arguments,
            and returns a SceneWriter instance. This factory will be called once
            per worker process to create a writer for that process.
        finalize : Callable[[SceneWriter], None], optional
            An optional function that takes a SceneWriter and performs any
            necessary finalization (e.g., closing files). If not provided, the
            `finalize` method of the SceneWriter will be called by default.

            However, if a finalize function is provided, it will be called
            instead of the default, which means that the SceneWriter's
            `finalize` method will not be called unless the provided function
            explicitly calls it.
        split_generator : Iterable[DatasetSplit], optional
            An optional generator that yields DatasetSplit values. If provided, a
            the values from this generator will synchronously fed to worker worker
            processes and passed into the `SceneWriter.write`.

        Practical Considerations
        ------------------------
        If provided the `split_generator` should at least provide as many splits
        as there are scenes. See `StreamSplitter` for a generic and configurable
        implementation of a split generator.

        Moreover if splits are provided using `StreamSplitter` the writer
        should not consume a split unless it is actually use, because this will
        result in a skewed distribution (different from the specified weights)
        of scenes across splits. For example, if the writer consumes a split for
        each scene, but some scenes are filtered out and not written, then the
        resulting distribution of scenes across splits will be different from
        the specified weights in `StreamSplitter`. In this case, it is
        recommended to only consume a split from the generator when a scene is
        actually written, which can be achieved by passing the split generator
        to the `SceneWriter.write` method and consuming splits there.

        """
        stop_event: threading.Event | None = None
        thread: threading.Thread | None = None
        if split_generator is not None:
            self._mp_split_queue = mp.Queue(maxsize=10)
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._feed_split_queue,
                args=(split_generator, self._mp_split_queue, stop_event),
                daemon=True,
            )
            thread.start()
        bound_factory = functools.partial(writer_factory)
        payloads = (_ProcessArgs(s, self._inner) for s in self._inner.sources())
        initargs = (*self._init_args, bound_factory, finalize)
        _ = deque(
            self._execute_parallel(self._process_fn_write, payloads, _init_write_worker, *initargs),
            maxlen=0,
        )

        if stop_event is not None and thread is not None:
            stop_event.set()
            thread.join(timeout=0)

        if self._mp_split_queue is not None:
            _ = self._mp_split_queue.get()
            self._mp_split_queue.close()
            self._mp_split_queue = None

    @staticmethod
    def _process_fn_write(args: _ProcessArgs[_S]) -> int:
        processed_scenes: int = 0

        def stream_split() -> Iterator[DatasetSplit]:
            if _split_queue is None:
                msg = "Split generator was not initialized for this worker process."
                raise ValueError(msg)
            while True:
                item = _split_queue.get()
                if item is None:
                    msg = "Split generator exhausted before writing completed."
                    raise ValueError(msg)
                yield item

        stream_split_iter = stream_split() if _split_queue is not None else None
        for scene in ParallelExecutor._generate_scenes(args.loader, args.source):
            _ = _writer.write(scene, splits=stream_split_iter)
            processed_scenes += 1

        with _source_counter.get_lock():
            _source_counter.value += 1

        return processed_scenes

    @staticmethod
    def _process_fn(args: _ProcessArgs[_S]) -> list[Scene]:
        """Worker process function that processes a single source and returns a list of Scenes.

        Parameters
        ----------
        args : _ProcessArgs[SourceT]
            Arguments containing the source and loader to process.

        Returns
        -------
        list[Scene]
            List of processed scenes from the given source.

        """
        scenes = list(ParallelExecutor._generate_scenes(args.loader, args.source))
        with _source_counter.get_lock():
            _source_counter.value += 1
        return scenes

    @staticmethod
    def _process_fn_callback(args: _ProcessArgs[_S]) -> int:
        """Worker process function that applies a callback to each Scene.

        This function returns the number of processed scenes (used for keeping
        track of progress). It behaves similarly to `_process_fn` but instead
        of returning the scenes, it applies the provided callable to each scene
        for side effects (e.g., saving to disk).

        Parameters
        ----------
        args : _ProcessArgs[SourceT]
            Arguments containing the source, loader, and callback to apply.

        Returns
        -------
        int
            Number of scenes processed from the given source.

        """
        processed_scenes = 0
        if args.fn is None:
            msg = "no callable function provided in _ProcessArgs."
            raise ValueError(msg)

        for scene in ParallelExecutor._generate_scenes(args.loader, args.source):
            _ = args.fn(scene)
            processed_scenes += 1

        with _source_counter.get_lock():
            _source_counter.value += 1

        return processed_scenes

    @staticmethod
    def _generate_scenes(loader: ProcessableLoader[_S], source: Source[_S]) -> Iterator[Scene]:
        """Core logic to process a source and yield properly numbered scenes."""
        for scene_data, map_resolver in loader.process_next(source):
            with _scene_counter.get_lock():
                _scene_counter.value += 1
                scene_number = _scene_counter.value
            yield loader.create_scene(scene_data, source, map_resolver, scene_number)

    def _execute_parallel(
        self,
        process_fn: Callable[[PayloadT], ReturnT],
        payloads: Iterable[PayloadT],
        initializer: Callable[P, object],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterable[ReturnT]:
        """Centralized execution engine for managing the Pool and progress tracking."""
        pool_initializer = functools.partial(initializer, *args, **kwargs)
        self._mp_scene_counter.value, self._mp_source_counter.value = 0, 0

        with (
            tqdm.tqdm(**self._tqdm_args()) as progress_bar,
            mp.Pool(self._processes, initializer=pool_initializer) as pool,
        ):
            map_fn = pool.imap if self._maintain_order else pool.imap_unordered
            work_iter = map_fn(process_fn, payloads, self._chunksize)

            yield from self._track_progress(work_iter, progress_bar)
            pool.close()
            pool.join()

    def _track_progress(
        self, work_iter: Iterable[ReturnT], progress_bar: tqdm.tqdm[NoReturn]
    ) -> Iterable[ReturnT]:
        if self._progress_bar == ProgressBar.NONE:
            yield from work_iter
            return

        for result in work_iter:
            self._update_progress_bar(progress_bar, result)
            yield result

    def _update_progress_bar(self, progress_bar: tqdm.tqdm[NoReturn], result: ReturnT) -> None:
        if self._progress_bar == ProgressBar.SOURCES:
            progress_bar.set_postfix_str(f"scenes: {self._mp_scene_counter.value}", refresh=False)
            _ = progress_bar.update(1)

        elif self._progress_bar == ProgressBar.SCENES:
            progress_bar.set_postfix_str(f"sources: {self._mp_source_counter.value}", refresh=False)
            processed_scenes = len(result) if isinstance(result, list) else result
            _ = progress_bar.update(processed_scenes)

    @staticmethod
    def _feed_split_queue(
        split_generator: Iterator[DatasetSplit],
        q: mp.Queue[DatasetSplit | None],
        stop_event: threading.Event,
    ) -> None:
        try:
            while not stop_event.is_set():
                split = next(split_generator)
                q.put(split)
        finally:
            q.put(None)

    def _total(self) -> int | None:
        if self._progress_bar == ProgressBar.SOURCES:
            return self._inner.num_sources()
        if self._progress_bar == ProgressBar.SCENES:
            return self._inner.num_scenes()
        return None

    def _tqdm_args(self) -> _TqdmArgs:
        return {
            "total": self._total(),
            "disable": self._progress_bar == ProgressBar.NONE,
            "unit": self._progress_bar.unit(),
            "desc": "Processing",
            "colour": "blue" if self._progress_bar == ProgressBar.SOURCES else "green",
            "unit_scale": self._progress_bar == ProgressBar.SCENES,
        }

    @staticmethod
    def _optimal_chunksize(num_sources: int | None, num_processes: int | None) -> int:
        if num_sources is None:
            return 1

        num_processes = num_processes or mp.cpu_count()
        chunksize, extra = divmod(num_sources, num_processes * 4)
        chunksize += int(extra > 0)
        return max(chunksize, 1)

    @property
    def _init_args(
        self,
    ) -> tuple[
        Synchronized[int],
        Synchronized[int],
        Synchronized[int],
        Lock,
        mp.Queue[DatasetSplit | None] | None,
    ]:
        """Convenience property to get the arguments needed for worker initialization."""
        return (
            self._mp_worker_counter,
            self._mp_scene_counter,
            self._mp_source_counter,
            self._mp_info_lock,
            self._mp_split_queue,
        )


class _ProcessArgs(NamedTuple, Generic[SourceT]):
    """Simplified arguments for processing a single source in a worker process."""

    source: Source[SourceT]
    loader: ProcessableLoader[SourceT]
    fn: Callable[..., object] | None = None


# Worker initialization function to set up the global counter in each worker
# process.
_worker_counter: Synchronized[int]
_scene_counter: Synchronized[int]
_source_counter: Synchronized[int]
_split_queue: mp.Queue[DatasetSplit | None] | None
_worker_id: int
_worker_info_lock: Lock
_writer: SceneWriter


@contextlib.contextmanager
def _get_worker_info() -> Generator[WorkerInfo, None, None]:
    with _worker_info_lock, _worker_counter.get_lock():
        yield WorkerInfo(
            worker_id=_worker_id,
            num_workers=_worker_counter.value,
            remaining_workers=_worker_counter.value,
        )


def _init_worker(
    worker_counter: Synchronized[int],
    scene_counter: Synchronized[int],
    source_counter: Synchronized[int],
    info_lock: Lock,
    split_queue: mp.Queue[DatasetSplit | None] | None,
) -> None:
    # This is the standard and most efficient way to share a counter across
    # processes in Python's multiprocessing module.
    global _worker_counter  # noqa: PLW0603
    _worker_counter = worker_counter
    global _worker_id  # noqa: PLW0603
    with worker_counter.get_lock():
        _worker_id = worker_counter.value
        worker_counter.value += 1
    global _scene_counter  # noqa: PLW0603
    _scene_counter = scene_counter
    global _source_counter  # noqa: PLW0603
    _source_counter = source_counter
    global _worker_info_lock  # noqa: PLW0603
    _worker_info_lock = info_lock
    global _split_queue  # noqa: PLW0603
    _split_queue = split_queue


def _init_write_worker(
    worker_counter: Synchronized[int],
    scene_counter: Synchronized[int],
    source_counter: Synchronized[int],
    info_lock: Lock,
    split_queue: mp.Queue[DatasetSplit | None] | None,
    writer_factory: Callable[[int], SceneWriter],
    finalize: Callable[[SceneWriter], None] | None,
) -> None:
    _init_worker(worker_counter, scene_counter, source_counter, info_lock, split_queue)

    global _writer  # noqa: PLW0603
    _writer = writer_factory(_worker_id)

    def cleanup() -> None:
        with _get_worker_info() as info:
            if finalize is not None:
                finalize(_writer)
            else:
                _writer.finish_local()

            if info.remaining_workers == 1:
                _writer.finish_final()

            _worker_counter.value -= 1

    _ = Finalize(obj=None, callback=cleanup, exitpriority=10)
