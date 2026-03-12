from __future__ import annotations

from typing import TYPE_CHECKING, Any, Concatenate

import tqdm
from typing_extensions import override

from dronalize.execution.common import ProgressBar
from dronalize.loading import ProcessableLoader, SceneLoader

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dronalize._internal._types import P
    from dronalize.categories import DatasetSplit
    from dronalize.loading import SceneWriter
    from dronalize.scene import Scene


class SequentialExecutor(SceneLoader):
    """Sequential scene processor that wraps a processable loader.

    It processes data strictly in the main thread. This avoids the overhead
    associated with Python's multiprocessing module, making it ideal for
    datasets with fast processing times, debugging, or environments where
    multiprocessing is not feasible.
    """

    def __init__(
        self,
        inner: ProcessableLoader[Any],
        *,
        progress_bar: ProgressBar | bool = ProgressBar.NONE,
    ) -> None:
        """Initialize the sequential processor.

        Parameters
        ----------
        inner : ProcessableLoader[SourceT]
            The underlying loader that discovers sources and creates scenes.
        progress_bar : ProgressBar | bool, optional
            Whether to show a progress bar and at what level (sources or scenes).

        """
        self._inner: ProcessableLoader[Any] = inner
        self._progress_bar: ProgressBar = (
            ProgressBar(int(progress_bar)) if isinstance(progress_bar, bool) else progress_bar
        )

    @override
    def scenes(self) -> Iterable[Scene]:
        """Process scenes sequentially and yield them one by one."""
        yield from self._generate_and_track()

    @override
    def scenes_callback(
        self,
        callback: Callable[Concatenate[Scene, P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Process scenes sequentially using a callback function."""
        for scene in self._generate_and_track():
            callback(scene, *args, **kwargs)

    @override
    def write_scenes(
        self,
        writer_factory: Callable[[], SceneWriter],
        finalize: Callable[[SceneWriter], None] | None = None,
        split_generator: Iterable[DatasetSplit] | None = None,
    ) -> None:
        """Process scenes sequentially and write them using a SceneWriter."""
        writer = writer_factory()
        split_iter = iter(split_generator) if split_generator is not None else None

        for scene in self._generate_and_track():
            _ = writer.write(scene, splits=split_iter)
        if finalize is not None:
            finalize(writer)
        else:
            writer.finish_local()
            writer.finish_final()

    def _total(self) -> int | None:
        if self._progress_bar == ProgressBar.SOURCES:
            return self._inner.num_sources()
        if self._progress_bar == ProgressBar.SCENES:
            return self._inner.num_scenes()
        return None

    def _tqdm_args(self) -> dict[str, Any]:
        return {
            "total": self._total(),
            "disable": self._progress_bar == ProgressBar.NONE,
            "unit": self._progress_bar.unit(),
            "desc": "Processing",
            "colour": "blue" if self._progress_bar == ProgressBar.SOURCES else "green",
            "unit_scale": True if self._progress_bar == ProgressBar.SCENES else None,
        }

    def _generate_and_track(self) -> Iterable[Scene]:
        """Core private generator handling both scene creation and progress tracking."""
        scene_counter = 0
        source_counter = 0

        with tqdm.tqdm(**self._tqdm_args()) as progress_bar:
            for source in self._inner.sources():
                for scene_data, map_resolver in self._inner.process_next(source):
                    scene_counter += 1
                    yield self._inner.create_scene(scene_data, source, map_resolver, scene_counter)

                    if self._progress_bar == ProgressBar.SCENES:
                        progress_bar.set_postfix_str(f"sources: {source_counter}", refresh=False)
                        _ = progress_bar.update(1)

                source_counter += 1
                if self._progress_bar == ProgressBar.SOURCES:
                    progress_bar.set_postfix_str(f"scenes: {scene_counter}", refresh=False)
                    _ = progress_bar.update(1)
