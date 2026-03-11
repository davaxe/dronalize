from __future__ import annotations

import threading
from collections.abc import Callable
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, Protocol

from dronalize.loading import SceneWriter

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dronalize.categories import DatasetSplit
    from dronalize.execution.common import Progress

AnyEvent = Event | threading.Event

WriterFactory = Callable[[int], SceneWriter]


class WritingExecutor(Protocol):
    def execute(
        self,
        writer_factory: WriterFactory,
        finalize: Callable[[SceneWriter], None] | None = None,
        split_generator: Iterator[DatasetSplit] | None = None,
    ) -> None: ...


class ObservableWritingExecutor(WritingExecutor, Protocol):
    def progress_event(self) -> AnyEvent:
        """Return an event that is set when the executor starts processing."""
        ...

    def is_running(self) -> bool:
        """Return whether the executor is currently running."""
        ...

    def progress(self) -> Progress:
        """Return the current progress of the executor."""
        ...
