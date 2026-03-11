from __future__ import annotations

import contextlib
import multiprocessing as mp
import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator
    from multiprocessing.sharedctypes import Synchronized
    from multiprocessing.synchronize import Event, Lock

    from dronalize.categories import DatasetSplit
    from dronalize.loading import SceneWriter


@dataclass(slots=True)
class WorkerRegistry:
    """Shared state for assigning unique worker IDs.

    This state is intentionally kept separate from progress accounting and
    optional split dispatching. It exists only to allocate stable worker-local
    integer IDs such as 0, 1, ..., n.
    """

    next_worker_id: Synchronized[int]

    @classmethod
    def create(cls) -> WorkerRegistry:
        """Create a new worker registry using the default multiprocessing context."""
        return cls(next_worker_id=mp.Value("i", 0))

    def reset(self) -> None:
        """Reset worker ID allocation back to zero."""
        with self.next_worker_id.get_lock():
            self.next_worker_id.value = 0

    def next_worker(self) -> int:
        """Return the next available worker ID."""
        with self.next_worker_id.get_lock():
            worker_id = self.next_worker_id.value
            self.next_worker_id.value += 1
            return worker_id


@dataclass(slots=True)
class ProgressState:
    """Shared counters used for reporting executor progress.

    Notes
    -----
    `processed_sources` counts sources whose processing function completed.
    `processed_scenes` counts scenes as they are created inside workers.

    The scene count reflects actual scene creation order across worker
    processes, which is generally not deterministic when multiprocessing is
    enabled.
    """

    active_workers: Synchronized[int]
    scene_counter: Synchronized[int]
    source_counter: Synchronized[int]
    snapshot_lock: Lock
    update_event: Event

    @classmethod
    def create(cls) -> ProgressState:
        """Create a new progress state using the default multiprocessing context."""
        return cls(
            active_workers=mp.Value("i", 0),
            scene_counter=mp.Value("i", 0),
            source_counter=mp.Value("i", 0),
            snapshot_lock=mp.Lock(),
            update_event=mp.Event(),
        )

    def reset(self) -> None:
        """Reset all progress counters back to zero."""
        with (
            self.active_workers.get_lock(),
            self.scene_counter.get_lock(),
            self.source_counter.get_lock(),
        ):
            self.active_workers.value = 0
            self.scene_counter.value = 0
            self.source_counter.value = 0
            self.update_event.clear()

    def increment_scene(self) -> int:
        """Increment the global scene counter and return the new value."""
        with self.scene_counter.get_lock():
            self.scene_counter.value += 1
            update = self.scene_counter.value

        self.update_event.set()
        return update

    def increment_source(self) -> int:
        """Increment the global source counter and return the new value."""
        with self.source_counter.get_lock():
            self.source_counter.value += 1
            update = self.source_counter.value
        self.update_event.set()
        return update

    def worker_started(self) -> None:
        """Record that a worker process became active."""
        with self.active_workers.get_lock():
            self.active_workers.value += 1
        self.update_event.set()

    def worker_stopped(self) -> int:
        """Record that a worker process stopped.

        Returns
        -------
        int
            The remaining number of active workers after the decrement.
        """
        with self.active_workers.get_lock():
            self.active_workers.value -= 1
            update = self.active_workers.value
        self.update_event.set()
        return update

    def active_worker_count(self) -> int:
        """Return the number of currently active worker processes."""
        with self.active_workers.get_lock():
            return self.active_workers.value


@dataclass(slots=True)
class SharedResources:
    """Top-level shared resources used by worker processes.

    This type is intentionally thin. It groups smaller, purpose-specific shared
    state objects rather than mixing all coordination into one mutable class.
    """

    registry: WorkerRegistry
    progress: ProgressState

    @classmethod
    def create(cls) -> SharedResources:
        """Create all shared resources required by the executor."""
        return cls(
            registry=WorkerRegistry.create(),
            progress=ProgressState.create(),
        )

    def reset(self) -> None:
        """Reset all shared counters and worker ID allocation."""
        self.registry.reset()
        self.progress.reset()


@dataclass(slots=True)
class SplitDispatchConfig:
    """Worker-visible configuration for split consumption.

    Attributes
    ----------
    queue : mp.Queue[DatasetSplit | None]
        Queue from which workers consume split values.
    """

    queue: mp.Queue[DatasetSplit | None]


@dataclass(slots=True)
class SplitDispatcher:
    """Main-process helper that feeds dataset splits to worker processes.

    This helper is used only by `write_scenes()` when a split generator is
    provided. The main process runs a lightweight background thread that pulls
    values from the split generator and pushes them into a multiprocessing
    queue. Workers lazily consume from that queue only when a writer actually
    requests a split.

    Design notes
    ------------
    - The input split source is converted to an iterator immediately.
    - When the iterator is exhausted, one sentinel (`None`) is enqueued per
      worker so that all blocked workers can observe exhaustion.
    - Shutdown is cooperative and avoids blocking indefinitely on queue
      operations by using timed `put()`.
    """

    queue: mp.Queue[DatasetSplit | None]
    worker_count: int
    stop_event: threading.Event
    thread: threading.Thread

    @classmethod
    def create(
        cls,
        split_source: Iterable[DatasetSplit],
        worker_count: int,
        *,
        maxsize: int | None = None,
    ) -> SplitDispatcher:
        """Create and start a split dispatcher.

        Parameters
        ----------
        split_source : Iterable[DatasetSplit]
            Source of split values. It will be converted to an iterator exactly
            once when the dispatcher is created.
        worker_count : int
            Number of worker processes that may consume from the queue.
        maxsize : int, optional
            Queue size. If omitted, defaults to `worker_count * 2`.

        Returns
        -------
        SplitDispatcher
            A started dispatcher instance.
        """
        split_iter = iter(split_source)
        q: mp.Queue[DatasetSplit | None] = mp.Queue(maxsize=maxsize or max(worker_count * 2, 1))
        stop_event = threading.Event()
        thread = threading.Thread(
            target=cls._feed_queue,
            args=(split_iter, q, worker_count, stop_event),
            daemon=True,
        )
        dispatcher = cls(
            queue=q,
            worker_count=worker_count,
            stop_event=stop_event,
            thread=thread,
        )
        thread.start()
        return dispatcher

    def config(self) -> SplitDispatchConfig:
        """Return the worker-visible queue configuration."""
        return SplitDispatchConfig(queue=self.queue)

    def close(self) -> None:
        """Stop the feeder thread and close the queue.

        This method is safe to call during normal completion as well as in
        exception paths.
        """
        self.stop_event.set()
        self.thread.join()

        self.queue.close()
        self.queue.cancel_join_thread()

    @staticmethod
    def _feed_queue(
        split_iter: Iterator[DatasetSplit],
        q: mp.Queue[DatasetSplit | None],
        worker_count: int,
        stop_event: threading.Event,
    ) -> None:
        """Feed split values into the multiprocessing queue.

        On normal iterator exhaustion, one sentinel per worker is enqueued so
        that workers blocked in `get()` can wake up and fail fast with a clear
        error.

        If shutdown is requested through `stop_event`, the feeder returns
        without trying to fully drain or signal the queue, because workers are
        expected to already be finished or terminating.
        """
        try:
            while not stop_event.is_set():
                split = next(split_iter)

                while not stop_event.is_set():
                    try:
                        q.put(split, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        except StopIteration:
            for _ in range(worker_count):
                while True:
                    try:
                        q.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        if stop_event.is_set():
                            return
                        continue


@dataclass(slots=True)
class WorkerRuntime:
    """Process-local runtime data initialized once per worker process.

    Attributes
    ----------
    shared : SharedResources
        Shared multiprocessing state used for progress and worker ID allocation.
    worker_id : int
        Stable integer ID assigned once to this worker process.
    writer : SceneWriter | None
        Optional writer used only by `write_scenes()`.
    split_dispatch : SplitDispatchConfig | None
        Optional split queue configuration used only by `write_scenes()`.
    """

    shared: SharedResources
    worker_id: int
    writer: SceneWriter | None = None
    split_dispatch: SplitDispatchConfig | None = None

    def active_workers(self) -> int:
        """Return the number of currently active worker processes."""
        return self.shared.progress.active_worker_count()

    @contextlib.contextmanager
    def progress_snapshot_lock(self) -> Generator[None, None, None]:
        """Lock used only for coarse progress snapshots or serialized finalize steps.

        This lock should not be confused with the per-counter locks on the
        shared progress state.
        """
        with self.shared.progress.snapshot_lock:
            yield
