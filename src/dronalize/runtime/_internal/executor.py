"""Internal executor protocols."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from multiprocessing.synchronize import Event
from typing import TYPE_CHECKING, Protocol

from dronalize.io.base import DatasetWriter

if TYPE_CHECKING:
    from dronalize.core.scene.model import Scene
    from dronalize.runtime._internal.state import Progress

AnyEvent = Event | threading.Event
WriterFactory = Callable[[int], DatasetWriter]


class ObservableExecutor(Protocol):
    def execute(
        self, writer_factory: WriterFactory, finalize: Callable[[DatasetWriter], None] | None = None
    ) -> None: ...

    def execute_yield(self) -> Iterator[Scene]: ...

    def progress_event(self) -> AnyEvent: ...

    def is_running(self) -> bool: ...

    def progress(self) -> Progress: ...
