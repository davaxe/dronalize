"""Single-process executor for internal runtime execution."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from dronalize.runtime._internal.executor import ObservableExecutor, WriterFactory
from dronalize.runtime._internal.state import Progress

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from dronalize.core.scene import Scene
    from dronalize.io.base import DatasetWriter
    from dronalize.processing.loading.base import BaseSceneLoader
    from dronalize.processing.loading.loader import Source
    from dronalize.runtime._internal.scene import SceneBuilder


class SequentialExecutor(ObservableExecutor):
    def __init__(
        self,
        loader: BaseSceneLoader[Any, Any],
        builder: SceneBuilder,
        sources: Iterable[Source[Any]],
        *,
        limit: int | None = None,
    ) -> None:
        self._loader: BaseSceneLoader[Any, Any] = loader
        self._builder: SceneBuilder = builder
        self._sources: Iterable[Source[Any]] = sources
        self._limit: int | None = limit
        self._scene_counter: int = 0
        self._source_counter: int = 0
        self._split_counts: dict[str, int] = {"unsplit": 0, "train": 0, "val": 0, "test": 0}
        self._total_sources: int | None = loader.num_sources()
        self._update_event: threading.Event = threading.Event()
        self._running: bool = False

    @override
    def execute(
        self, writer_factory: WriterFactory, finalize: Callable[[DatasetWriter], None] | None = None
    ) -> None:
        writer = writer_factory(0)
        for scene in self._generate_and_track():
            writer.write(scene)
        if finalize is not None:
            finalize(writer)
        else:
            writer.finish_local()
            writer.finish_final()

    @override
    def execute_yield(self) -> Iterator[Scene]:
        yield from self._generate_and_track()

    @override
    def progress(self) -> Progress:
        return Progress(
            running=self._running,
            processed_sources=self._source_counter,
            processed_scenes=self._scene_counter,
            total_sources=self._total_sources,
            total_scenes=self._limit,
            active_workers=1 if self._running else 0,
            split_counts=self.split_counts,
        )

    @override
    def progress_event(self) -> threading.Event:
        return self._update_event

    @override
    def is_running(self) -> bool:
        return self._running

    @property
    def split_counts(self) -> dict[str, int]:
        return dict(self._split_counts)

    def _generate_and_track(self) -> Iterable[Scene]:
        remaining = self._limit
        self._running = True
        self._update_event.set()
        for source in self._sources:
            for processed in self._builder.prepare_source(self._loader, source):
                if remaining is not None:
                    if remaining == 0:
                        break
                    remaining -= 1
                self._update_event.set()
                scene = self._builder.create_scene(
                    self._loader, processed, source, self._scene_counter
                )
                self._scene_counter += 1
                split_key = (
                    "unsplit" if scene.split_assignment is None else scene.split_assignment.value
                )
                self._split_counts[split_key] += 1
                yield scene
            else:
                self._source_counter += 1
                self._update_event.set()
                continue
            break
        self._running = False
        self._update_event.set()
