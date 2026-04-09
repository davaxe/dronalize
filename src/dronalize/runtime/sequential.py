"""Single-process executor for sequential scene generation and writing."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from dronalize.runtime.executor import ObservableExecutor, Progress, WriterFactory

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dronalize.core.scene import Scene
    from dronalize.io.backends.base import DatasetWriter
    from dronalize.processing.loading.base import BaseSceneLoader


class SequentialExecutor(ObservableExecutor):
    """Sequential writing executor backed by a single processable loader.

    It processes data strictly in the main thread. This avoids the overhead
    associated with Python's multiprocessing module, making it ideal for
    datasets with fast processing times, debugging, or environments where
    multiprocessing is not feasible.
    """

    def __init__(self, inner: BaseSceneLoader[Any, Any], *, limit: int | None = None) -> None:
        """Initialize the sequential executor.

        Parameters
        ----------
        inner : DatasetLoader[SourceT]
            The underlying loader that discovers sources and creates scenes.
        limit : int, optional
            An optional limit on the total number of scenes to process. If `None`,
            all scenes from the loader will be processed.

        """
        self._inner: BaseSceneLoader[Any, Any] = inner
        self._limit: int | None = limit
        self._scene_counter: int = 0
        self._source_counter: int = 0
        self._total_sources: int | None = inner.num_sources()
        self._update_event: threading.Event = threading.Event()
        self._running: bool = False

    @override
    def execute(
        self, writer_factory: WriterFactory, finalize: Callable[[DatasetWriter], None] | None = None
    ) -> None:
        """Process all scenes sequentially and write them with worker ID `0`."""
        writer = writer_factory(0)

        for scene in self._generate_and_track():
            _ = writer.write(scene)
        if finalize is not None:
            finalize(writer)
        else:
            writer.finish_local()
            writer.finish_final()

    @override
    def progress(self) -> Progress:
        """Return the current sequential execution progress."""
        return Progress(
            running=self._running,
            processed_sources=self._source_counter,
            processed_scenes=self._scene_counter,
            total_sources=self._total_sources,
            total_scenes=self._limit,
            active_workers=1,
        )

    @override
    def progress_event(self) -> threading.Event:
        """Return the event used to signal start, updates, and completion."""
        return self._update_event

    @override
    def is_running(self) -> bool:
        """Return whether the sequential executor is actively processing."""
        return self._running

    def _generate_and_track(self) -> Iterable[Scene]:
        """Yield scenes while maintaining counters and progress notifications."""
        inner: BaseSceneLoader = self._inner
        remaining: int | None = self._limit
        self._running = True
        self._update_event.set()
        for source in inner.sources():
            for processed in inner.process_next(source):
                if remaining is not None:
                    if remaining == 0:
                        break
                    remaining -= 1
                self._update_event.set()
                yield (inner.create_scene(processed, source, self._scene_counter))
                self._scene_counter += 1
            else:
                self._source_counter += 1
                self._update_event.set()
                continue
            break

        self._running = False
        self._update_event.set()
