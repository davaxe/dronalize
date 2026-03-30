from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from typing import Protocol

from dronalize.io.writers.base import SceneWriter

AnyEvent = Event | threading.Event

WriterFactory = Callable[[int], SceneWriter]
"""Factory that creates a worker-local writer for a given worker ID."""


@dataclass(frozen=True, slots=True)
class Progress:
    """Snapshot of the executor's current progress."""

    running: bool
    processed_sources: int
    processed_scenes: int
    total_sources: int | None
    total_scenes: int | None
    active_workers: int


class WritingExecutor(Protocol):
    """Protocol for executing a scene-writing run.

    Implementations consume sources from a loader, build scenes, and write
    them through a worker-local `SceneWriter` created by `writer_factory`.
    """

    def execute(
        self, writer_factory: WriterFactory, finalize: Callable[[SceneWriter], None] | None = None
    ) -> None:
        """Run the executor and write all produced scenes.

        Parameters
        ----------
        writer_factory : WriterFactory
            Factory used to construct one `SceneWriter` per worker.
        finalize : Callable[[SceneWriter], None] | None, optional
            Optional writer finalization hook. When omitted, the executor uses
            the writer's default `finish_local()` and `finish_final()` methods.
        """
        ...


class ObservableWritingExecutor(WritingExecutor, Protocol):
    """Writing executor that can be observed."""

    def progress_event(self) -> AnyEvent:
        """Return an event set whenever execution state or progress may have changed."""
        ...

    def is_running(self) -> bool:
        """Return whether the executor is currently executing work."""
        ...

    def progress(self) -> Progress:
        """Return the current execution progress snapshot."""
        ...
