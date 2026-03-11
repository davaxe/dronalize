from __future__ import annotations

import itertools
import threading
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from dronalize.execution.common import Progress
from dronalize.execution.executor import ObservableWritingExecutor, WriterFactory

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from dronalize.categories import DatasetSplit
    from dronalize.loading import ProcessableLoader, SceneWriter
    from dronalize.scene import Scene


class SequentialExecutor(ObservableWritingExecutor):
    """Sequential scene processor that wraps a processable loader.

    It processes data strictly in the main thread. This avoids the overhead
    associated with Python's multiprocessing module, making it ideal for
    datasets with fast processing times, debugging, or environments where
    multiprocessing is not feasible.
    """

    def __init__(self, inner: ProcessableLoader[Any], *, limit: int | None = None) -> None:
        """Initialize the sequential processor.

        Parameters
        ----------
        inner : ProcessableLoader[SourceT]
            The underlying loader that discovers sources and creates scenes.

        """
        self._inner: ProcessableLoader[Any] = inner
        self._limit: int | None = limit
        self._scene_counter: int = 0
        self._source_counter: int = 0
        self._total_sources: int | None = inner.num_sources()
        self._total_scenes: int | None = inner.num_scenes()
        self._update_event: threading.Event = threading.Event()
        self._running: bool = False
        if self._limit and self._total_sources is not None:
            self._total_sources = min(self._total_sources, self._limit)

    @override
    def execute(
        self,
        writer_factory: WriterFactory,
        finalize: Callable[[SceneWriter], None] | None = None,
        split_generator: Iterator[DatasetSplit] | None = None,
    ) -> None:
        """Process scenes sequentially and write them using a SceneWriter."""
        writer = writer_factory(0)
        split_iter = iter(split_generator) if split_generator is not None else None

        for scene in self._generate_and_track():
            _ = writer.write(scene, splits=split_iter)
        if finalize is not None:
            finalize(writer)
        else:
            writer.finish_local()
            writer.finish_final()

    @override
    def progress(self) -> Progress:
        """Return the current progress of the executor."""
        return Progress(
            running=self._running,
            processed_sources=self._source_counter,
            processed_scenes=self._scene_counter,
            total_sources=self._total_sources,
            total_scenes=self._total_scenes,
            active_workers=1,
        )

    @override
    def progress_event(self) -> threading.Event:
        return self._update_event

    @override
    def is_running(self) -> bool:
        return self._running

    def _generate_and_track(self) -> Iterable[Scene]:
        """Core private generator handling both scene creation and progress tracking."""
        self._running = True
        self._update_event.set()
        for source in itertools.islice(self._inner.sources(), self._limit):
            for scene_data, map_resolver in self._inner.process_next(source):
                self._scene_counter += 1
                self._update_event.set()
                yield self._inner.create_scene(
                    scene_data, source, map_resolver, self._scene_counter
                )
            self._source_counter += 1
            self._update_event.set()

        self._running = False
        self._update_event.set()
