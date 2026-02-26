from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import replace
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar

import tqdm
from typing_extensions import Self, override

from preprocessing.core.interface import BaseSceneLoader, LoaderConfig, Scene

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

    import polars as pl


class ProgressBar(IntEnum):
    """Enum to specify which progress bars to show when using ParallelLoader."""

    NONE = 0
    """No progress bars."""
    SCENES = 1
    """Show progress bar for scenes."""
    SOURCES = 2
    """Show progress bar for sources."""

    def unit(self) -> str:
        """Return the unit string to use for tqdm progress bars based on the selected ProgressBar."""
        if self == ProgressBar.SOURCES:
            return " sources"
        if self == ProgressBar.SCENES:
            return " scenes"
        return ""


T_Source = TypeVar("T_Source")
T_ID = TypeVar("T_ID", bound=(Hashable))


class ParallelLoader(BaseSceneLoader[T_ID, T_Source]):
    """A generic parallel data processor that wraps another DataProcessor.

    It uses Python's multiprocessing module to parallelize the processing of sources across multiple
    CPU cores. The number of processes and chunksize can be configured depending on workload.

    By default the order resulting scenes when using `scenes_iter` method will not be deterministic
    across runs or systems due to the use of `imap_unordered` for better performance. If a
    deterministic order is required, the `maintain_order` flag can be set to True, which will use `imap`
    instead of `imap_unordered`.

    This class also implements the `SceneLoader` interface, which means it can be used as a drop in
    replacement for any other `SceneLoader` implementation.

    Practical Considerations:
        Using multprocessing can significantly speed up the processing of large datasets, however
        there are som important considerations to keep in mind:

            1. If the underlying sources are small and quick to process, the overhead of multiprocessing
               may outweigh the benefits. One option is to increase the `chunksize` to reduce overhead
               in this case.

            2. If sources are instead large, for example `WaymoLoader` and its tfrecord files,
               then other issues may arise such as increased memory usage and possibly issues
               with inter-process comumication. In these cases it is a good idea to avoid `scenes_iter`
               method, and instead use the `process_scenes` method with a callback function that
               performs side effects (e.g., saving to file). If an error related to "Too many opened
               files" is encountered when using `scenes_iter`, this is a strong indication that the
               underlying sources are too large for `scenes_iter` and `process_scenes` should be
               used.

            3. Since `DataLoaders` utilizes Polars, which also uses multiple cores for processing,
               there may be some contention for CPU resources between the multiprocessing in `ParallelLoader`
               and the multithreading in Polars. Setting the environment variable `POLARS_MAX_THREADS=X`,
               where X is a number that balances the workload between `ParallelLoader` and Polars.
               An alternative option is to lower the `processes` parameter in this class.
    """

    def __init__(
        self,
        processor: BaseSceneLoader[T_ID, T_Source],
        chunksize: int = 1,
        processes: int | None = None,
        *,
        maintain_order: bool = False,
        progress_bar: ProgressBar | bool = ProgressBar.NONE,
    ) -> None:
        """Initialize with the given processor and multiprocessing configuration.

        Args:
            processor: The underlying DataProcessor to use for processing each source.
            chunksize: The number of sources to process in each batch when using multiprocessing.
                Larger chunksizes can reduce overhead but may lead to less even load distribution
                across processes.
            processes: The number of processes to use for multiprocessing. If None, uses the
                number of CPU cores.
            maintain_order: If True, the order of resulting scenes from `scenes_iter` will be
                computed using `imap` to maintain the order of sources.
            progress_bar: Whether to show a progress bar using tqdm. Can be set to:
                - `ProgressBar.NONE` (default): No progress bar.
                - `ProgressBar.SOURCES`: Show progress bar for sources.
                - `ProgressBar.SCENES`: Show progress bar for scenes.

        """
        if processes is not None and processes <= 1:
            msg = "number of processes must be greater than 1 for ParallelLoader."
            raise ValueError(msg)

        self._processor = processor
        self._mp_scene_counter: Synchronized[int] = mp.Value("i", 0)
        self._mp_source_counter: Synchronized[int] = mp.Value("i", 0)
        self._chunksize: int = chunksize
        self._processes: int | None = processes
        self._maintain_order: bool = maintain_order
        self._progress_bar: ProgressBar = (
            ProgressBar(int(progress_bar)) if isinstance(progress_bar, bool) else progress_bar
        )

    def processes(self, processes: int | None) -> Self:
        """Set the number of processes to use for multiprocessing. If None, uses the number of CPU cores."""
        self._processes = processes
        return self

    def chunksize(self, chunksize: int) -> Self:
        """Set chunksize used when creating the `multiprocessing.Pool`.

        Larger chunksizes can reduce overhead but may lead to less even load distribution across
        processes.
        """
        self._chunksize = chunksize
        return self

    @override
    def sources(self) -> Iterable[tuple[T_ID, T_Source]]:
        return self._processor.sources()

    @override
    def load_raw(self, source: T_Source) -> Iterable[pl.LazyFrame]:
        return self._processor.load_raw(source)

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return self._processor.normalize(df)

    @override
    def default_config(self) -> LoaderConfig:
        return self._processor.default_config()

    @override
    def scenes(self) -> Iterable[Scene[T_ID]]:
        """Process scenes in parallel and yield them one by one.

        This uses `multiprocessing` to process data in parallel. It works by creating a pool of
        worker processes that process batches (chunks, determined by `chunksize`) of sources.
        The results are collected and yielded as scenes.

        Note that each source might yield multiple scenes (for example if windowing is used),
        and the each worker will process all scenes before returning to the main processes.
        """
        with (
            tqdm.tqdm(**self._tqdm_args()) as progress_bar,
            mp.Pool(
                self._processes,
                initializer=_init_worker,
                initargs=(self._mp_scene_counter, self._mp_source_counter),
            ) as pool,
        ):
            map_fn = pool.imap if self._maintain_order else pool.imap_unordered
            for scenes in map_fn(
                ParallelLoader._process_fn,
                (_ProcessArgs(s, self._processor) for s in self.sources()),
                self._chunksize,
            ):
                if self._progress_bar == ProgressBar.SOURCES:
                    progress_bar.update(1)
                elif self._progress_bar == ProgressBar.SCENES:
                    progress_bar.update(len(scenes))

                yield from scenes

    def scenes_callback(self, callback: Callable[[Scene[T_ID]], None]) -> None:
        """Process scenes using a callback function instead of yielding them.

        This method allows you to provide a callback function that will be called for each processed
        scene. This can be more efficient than yielding scenes if you want to perform side effects
        (e.g., saving to disk) without needing to store all scenes in memory at once.

        Args:
            callback: A callable that takes a Scene as input and performs side effects (e.g., saving
            to disk). This function will be called for each processed scene.

        """
        with (
            tqdm.tqdm(**self._tqdm_args()) as progress_bar,
            mp.Pool(
                self._processes,
                initializer=_init_worker,
                initargs=(self._mp_scene_counter, self._mp_source_counter),
            ) as pool,
        ):
            map_fn = pool.imap if self._maintain_order else pool.imap_unordered
            work_iter = map_fn(
                ParallelLoader._process_fn_callable,
                (_ProcessArgs(s, self._processor, fn=callback) for s in self._processor.sources()),
                self._chunksize,
            )
            if self._progress_bar == ProgressBar.NONE:
                # Exaust the iterator
                deque(work_iter, maxlen=0)
                return

            for processed_scenes in work_iter:
                if self._progress_bar == ProgressBar.SOURCES:
                    progress_bar.update(1)
                elif self._progress_bar == ProgressBar.SCENES:
                    progress_bar.update(processed_scenes)

    @staticmethod
    def _process_fn(args: _ProcessArgs[T_ID, T_Source]) -> list[Scene[T_ID]]:
        """Worker process function that processes a single source and returns a list of Scenes."""
        processor = args.processor
        scenes: list[Scene[T_ID]] = []
        scene_number: int = -1
        for scene_data in processor.process_next(args.source[1]):
            with _scene_counter.get_lock():
                _scene_counter.value += 1
                scene_number = _scene_counter.value

            scene = processor.create_scene(scene_data, args.source[0])
            # Add a unique global scene number, since each worker will have its own local counter.
            scenes.append(replace(scene, scene_number=scene_number))

        with _source_counter.get_lock():
            _source_counter.value += 1
        return scenes

    @staticmethod
    def _process_fn_callable(args: _ProcessArgs[T_ID, T_Source]) -> int:
        """Worker process function that applies a callable to each Scene.

        This function returns the number of processed scenes (used for keeping track of progress),
        otherwise it behaves similarly to `_process_fn` but instead of returning the scenes, it
        applies the provided callable to each scene for side effects (e.g., saving to disk).
        """
        if args.fn is None:
            msg = "no callable function provided in _ProcessArgs for _process_fn_callable."
            raise ValueError(msg)

        processed_scenes = 0
        processor = args.processor
        scene_number: int = -1
        for scene_data in processor.process_next(args.source[1]):
            with _scene_counter.get_lock():
                _scene_counter.value += 1
                scene_number = _scene_counter.value

            scene = processor.create_scene(scene_data, args.source[0])
            scene = replace(scene, scene_number=scene_number)
            args.fn(scene)
            processed_scenes += 1

        with _source_counter.get_lock():
            _source_counter.value += 1

        return processed_scenes

    def _total(self) -> int | None:
        if self._progress_bar == ProgressBar.SOURCES:
            return self._processor.num_sources()
        if self._progress_bar == ProgressBar.SCENES:
            return self._processor.num_scenes()
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


class _ProcessArgs(NamedTuple, Generic[T_ID, T_Source]):
    """Arguments for processing a single source in a worker process."""

    source: tuple[T_ID, T_Source]
    processor: BaseSceneLoader[T_ID, T_Source]
    fn: Callable[[Scene[T_ID]], None] | None = None


# Worker initialization function to set up the global counter in each worker process.
_scene_counter: Synchronized[int]
_source_counter: Synchronized[int]


def _init_worker(scene_counter: Synchronized[int], source_counter: Synchronized[int]) -> None:
    # This is the standard and most efficient way to share a counter across processes in Python's
    # multiprocessing module.
    global _scene_counter  # noqa: PLW0603
    _scene_counter = scene_counter
    global _source_counter  # noqa: PLW0603
    _source_counter = source_counter


if __name__ == "__main__":
    from pathlib import Path

    from preprocessing.datasets.waymo.loader import WaymoLoader

    directory = Path("/home/west/Developer/behavior-prediction/datasets/waymo/validation")

    processor_single = WaymoLoader(
        directory,
        "*validation.tfrecord*",
        include_map=True,
        min_distance=2,
    )
    processor = ParallelLoader(
        processor_single,
        maintain_order=False,
        chunksize=1,
        progress_bar=ProgressBar.SCENES,
    )
    start_time = time.perf_counter()
    deque(processor.scenes(), maxlen=0)  # Exhaust the generator to process all scenes.
    multi_time = time.perf_counter() - start_time
    print(
        f"Processed all scenes ({processor._mp_scene_counter.value}) in {multi_time:.2f} seconds with multiprocessing.",
    )

    start_time = time.perf_counter()
    for _scene in tqdm.tqdm(
        processor_single.scenes(),
        total=processor._mp_scene_counter.value,
        desc="Processing scenes without multiprocessing",
        colour="red",
        unit=" scenes",
    ):
        continue

    single_time = time.perf_counter() - start_time
    print(
        f"Processed all scenes ({processor_single._count}) in {single_time:.2f} seconds without multiprocessing.",
    )

    print(f"Multiprocessing speedup: {single_time / multi_time:.2f}x")
