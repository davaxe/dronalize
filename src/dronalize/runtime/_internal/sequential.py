"""Single-process executor for internal runtime execution."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.runtime._internal.executor import Executor, WriterFactory
from dronalize.runtime._internal.state import Progress, SplitCounts

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from dronalize.core.scene import Scene
    from dronalize.runtime._internal.processor import RuntimeProcessor


class SequentialExecutor(Executor):
    """Single-process executor for internal runtime execution.

    Parameters
    ----------
    processor: RuntimeProcessor
        The runtime processor to execute, containing logic and cofigurations.
    limit: int | None, optional
        An optional limit on the total number of scenes to select. If None, no
        limit will be applied. Default is None.
    """

    def __init__(self, processor: RuntimeProcessor, *, limit: int | None = None) -> None:
        self._processor: RuntimeProcessor = processor
        self._limit: int | None = limit
        self._candidate_scene_counter: int = 0
        self._selected_scene_counter: int = 0
        self._source_counter: int = 0
        self._split_counts: SplitCounts = {"unsplit": 0, "train": 0, "val": 0, "test": 0}
        self._screening_enabled: bool = processor.screening_enabled()
        self._total_sources: int | None = processor.total_sources()
        self._update_event: threading.Event = threading.Event()
        self._running: bool = False

    @override
    def execute(self, writer_factory: WriterFactory) -> None:
        writer = writer_factory(0)
        try:
            for scene in self._generate_and_track():
                writer.write(scene)
        finally:
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
            candidate_scenes=self._candidate_scene_counter,
            selected_scenes=self._selected_scene_counter,
            total_sources=self._total_sources,
            scene_limit=self._limit,
            active_workers=1 if self._running else 0,
            split_counts=self._split_counts,
            screening_enabled=self._screening_enabled,
        )

    @override
    def progress_event(self) -> threading.Event:
        return self._update_event

    @override
    def is_running(self) -> bool:
        return self._running

    def _generate_and_track(self) -> Iterable[Scene]:
        self._running = True
        self._update_event.set()
        for source in self._processor.iter_sources():
            if self._selected_scene_limit_reached():
                break
            self._source_counter += 1
            self._update_event.set()
            for candidate in self._processor.iter_candidates(source):
                self._record_candidate_scene()
                if not candidate.passes_screening:
                    continue
                scene_number = self._claim_selected_scene()
                if scene_number is None:
                    self._running = False
                    self._update_event.set()
                    return
                scene = self._processor.materialize(candidate, scene_number)
                split_key = (
                    "unsplit" if scene.split_assignment is None else scene.split_assignment.value
                )
                self._split_counts[split_key] += 1
                self._update_event.set()
                yield scene
        self._running = False
        self._update_event.set()

    def _record_candidate_scene(self) -> None:
        self._candidate_scene_counter += 1
        self._update_event.set()

    def _claim_selected_scene(self) -> int | None:
        if self._selected_scene_limit_reached():
            return None
        scene_number = self._selected_scene_counter
        self._selected_scene_counter += 1
        self._update_event.set()
        return scene_number

    def _selected_scene_limit_reached(self) -> bool:
        return self._limit is not None and self._selected_scene_counter >= self._limit
