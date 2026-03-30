from __future__ import annotations

import contextlib
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing.sharedctypes import Synchronized
    from multiprocessing.synchronize import Event, Lock

    from dronalize.io.writers.base import SceneWriter


@dataclass(slots=True)
class WorkerRegistry:
    """Shared state for assigning unique worker IDs.

    This state is intentionally kept separate from progress accountin. It exists
    only to allocate stable worker-local integer IDs such as 0, 1, ..., n.

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
        return cls(registry=WorkerRegistry.create(), progress=ProgressState.create())

    def reset(self) -> None:
        """Reset all shared counters and worker ID allocation."""
        self.registry.reset()
        self.progress.reset()


QueueItem = tuple[tuple[int | str, ...], Connection] | None


@dataclass(slots=True)
class WorkerRuntime:
    """Process-local runtime data initialized once per worker process."""

    shared: SharedResources
    """Shared multiprocessing state used for progress and worker ID allocation."""
    worker_id: int
    """Stable integer ID assigned once to this worker process."""
    writer: SceneWriter | None = None
    """Optional writer used only by `execute()`."""

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
