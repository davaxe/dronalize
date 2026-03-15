from __future__ import annotations

import threading
from collections.abc import Callable
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, Protocol

from dronalize.loading import SceneWriter

if TYPE_CHECKING:
    from dronalize.execution.assigner import SplitAssigner
    from dronalize.execution.common import Progress

AnyEvent = Event | threading.Event

WriterFactory = Callable[[int], SceneWriter]
"""Factory that creates a worker-local writer for a stable integer worker ID."""


class WritingExecutor(Protocol):
    """Protocol for executing a scene-writing run.

    Implementations consume sources from a loader, build scenes, and write
    them through a worker-local `SceneWriter` created by `writer_factory`.
    """

    def execute(
        self,
        writer_factory: WriterFactory,
        finalize: Callable[[SceneWriter], None] | None = None,
        split_assigner: SplitAssigner | None = None,
    ) -> None:
        """Run the executor and write all produced scenes.

        Parameters
        ----------
        writer_factory : WriterFactory
            Factory used to construct one `SceneWriter` per worker.
        finalize : Callable[[SceneWriter], None] | None, optional
            Optional writer finalization hook. When omitted, the executor uses
            the writer's default `finish_local()` and `finish_final()` methods.
        split_assigner : SplitAssigner | None, optional
            Optional split assigner. If given, the executor uses it to determine
            the split for each produced scene.
        """
        ...


class ObservableWritingExecutor(WritingExecutor, Protocol):
    """Writing executor that can be observed by a progress reporter."""

    def progress_event(self) -> AnyEvent:
        """Return an event set whenever execution state or progress may have changed."""
        ...

    def is_running(self) -> bool:
        """Return whether the executor is currently executing work."""
        ...

    def progress(self) -> Progress:
        """Return the current execution progress snapshot."""
        ...
