from __future__ import annotations

import multiprocessing as mp
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import replace
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Concatenate, Generic, NamedTuple, ParamSpec, TypeVar

import tqdm
from typing_extensions import Self, override

from preprocessing.core.protocols.loader import BaseSceneLoader, SceneLoader, Source

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

    from preprocessing.core.datatypes.scene import Scene


class ProgressBar(IntEnum):
    """Enum to specify which progress bars to show when using ParallelSceneLoader."""

    NONE = 0
    """No progress bars."""
    SOURCES = 1
    """Show progress bar for sources (# processed scenes are shown i postfix)."""
    SCENES = 2
    """Show progress bar for scenes (# processed sources are shown in postfix)."""

    def unit(self) -> str:
        """Return the unit string to use for tqdm progress bars based on the selected ProgressBar."""
        if self == ProgressBar.SOURCES:
            return " sources"
        if self == ProgressBar.SCENES:
            return " scenes"
        return ""


T_Source = TypeVar("T_Source")
T_ID = TypeVar("T_ID", bound=(Hashable))
P = ParamSpec("P")


class ParallelSceneLoader(SceneLoader[T_ID]):
    """A generic parallel data loader that wraps another BaseSceneLoader.

    It uses Python's multiprocessing module to parallelize the processing of sources across multiple
    CPU cores. The number of processes and chunksize can be configured depending on workload.

    By default the order resulting scenes when using `scenes` method will not be deterministic
    across runs or systems due to the use of `imap_unordered` for better performance. If a
    deterministic order is required, the `maintain_order` flag can be set to True, which will use
    `imap` instead of `imap_unordered`.

    Practical Considerations:
        Using multprocessing can significantly speed up the processing of large datasets, however
        there are som important considerations to keep in mind:

        1. If the underlying sources are small and quick to process, the overhead of multiprocessing
        may outweigh the benefits. One option is to increase the `chunksize` to reduce overhead in
        this case.

        2. If sources are instead large, for example `WaymoLoader` and its tfrecord files, then
        other issues may arise such as increased memory usage and possibly issues with inter-process
        comumication. In these cases it is a good idea to avoid `scenes` method, and instead use the
        `scenes_callback` method with a callback function that performs side effects (e.g., saving
        to file). If an error related to "Too many opened files" is encountered when using `scenes`,
        this is a strong indication that the underlying sources are too large for `scenes` and
        `scenes_callback` should be used.

        3. If the Scenes include a lot of data (e.g. large DataFrames or maps) sending them between
        processees can be costly, and `scenes_callback` should be preffered when possible. In
        extreme cases using multiprocessing can slow down processing due to the overhead.

        4. Since `DataLoaders` utilizes Polars, which also uses multiple cores for processing, there
        may be some contention for CPU resources between the multiprocessing in
        `ParallelSceneLoader` and the multithreading in Polars. Setting the environment variable
        `POLARS_MAX_THREADS=X`, where X is a number that balances the workload between
        `ParallelSceneLoader` and Polars. An alternative option is to lower the `processes`
        parameter in this class.

        5. The input dataloader will be sent to each worker process. This means that it should be
        as lightweight as possible, and possibly avoid eager loading of data in its constructor.
        It also needs to be picklable, in order to be sent to worker processes.

    """

    def __init__(
        self,
        inner: BaseSceneLoader[T_ID, T_Source],
        chunksize: int | None = None,
        processes: int | None = None,
        *,
        maintain_order: bool = False,
        progress_bar: ProgressBar | bool = ProgressBar.NONE,
    ) -> None:
        """Initialize with the given loader and multiprocessing configuration.

        Progress Bar Behavior:
            - If `progress_bar` is set to `ProgressBar.SOURCES`, a progress bar
            will be shown that tracks the number of sources processed.
            - If `progress_bar` is set to `ProgressBar.SCENES`, a progress bar will be shown that
            tracks the number of scenes processed.

        No matter what option is chosen, the other quantity (sources or scenes) will be tracked in
        he progress bar's postfix for visibility. Using `ProgressBar.SOURCES` is generally
        recomended since the total number of sources is often known beforehand, while number of
        scenes is usually not known beforehand (this result in only showng the number of processed
        scenes without a total, which can be less useful).

        Args:
            inner: The underlying BaseSceneLoader to use for processing each source.
            chunksize: The number of sources to process in each batch when using multiprocessing.
                Larger chunksizes can reduce overhead but may lead to less even load distribution
                across processes. If `None` tries to automatically determine an optimal chunksize
                based on the number of sources and processes (when possible),
            processes: The number of processes to use for multiprocessing. If None, uses the
                number of CPU cores.
            maintain_order: If True, the order of resulting scenes from `scenes` will be
                computed using `imap` to maintain the order of sources.
            progress_bar: Whether to show a progress bar using tqdm. `True` is equivalent to
                `ProgressBar.SOURCES`. Se behaviour description above.

        """
        if processes is not None and processes <= 1:
            msg = "number of processes must be greater than 1 for ParallelSceneLoader."
            raise ValueError(msg)

        self._inner = inner
        self._mp_scene_counter: Synchronized[int] = mp.Value("i", 0)
        self._mp_source_counter: Synchronized[int] = mp.Value("i", 0)
        self._chunksize: int = chunksize or self._optimal_chunksize(inner.num_sources(), processes)
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
                ParallelSceneLoader._process_fn,
                (_ProcessArgs(s, self._inner) for s in self._inner.sources()),
                self._chunksize,
            ):
                if self._progress_bar == ProgressBar.SOURCES:
                    progress_bar.set_postfix(
                        {"scenes": self._mp_scene_counter.value}, refresh=False
                    )
                    progress_bar.update(1)
                elif self._progress_bar == ProgressBar.SCENES:
                    progress_bar.set_postfix(
                        {"sources": self._mp_source_counter.value}, refresh=False
                    )
                    progress_bar.update(len(scenes))

                yield from scenes

    @override
    def scenes_callback(
        self,
        callback: Callable[Concatenate[Scene[T_ID], P], None],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Process scenes using a callback function instead of yielding them.

        This method allows you to provide a callback function that will be called for each processed
        scene. This can be more efficient than yielding scenes if you want to perform side effects
        (e.g., saving to disk) without needing to store all scenes in memory at once.

        Args:
            callback: A callable that takes a Scene as input and performs side effects (e.g., saving
                to disk). This function will be called for each processed scene.
            *args: Additional positional arguments to pass to the callback function.
            **kwargs: Additional keyword arguments to pass to the callback function.


        Considerations:
            All additional arguments will be passed to the worker processes and should be chosen
            carefully to avoid sending large amounts of data.

            Moreover, due to the nature of multiprocessing, the callback function and its arguments
            must be picklable. If you encounter issues with pickling, consider using simpler data
            structures or functions defined at the top level of a module.

        """
        loader = self._inner
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
                ParallelSceneLoader._process_fn_callback,
                (_ProcessArgs(s, loader, callback, args, kwargs) for s in loader.sources()),
                self._chunksize,
            )
            if self._progress_bar == ProgressBar.NONE:
                # Exaust the iterator
                deque(work_iter, maxlen=0)
                return

            for processed_scenes in work_iter:
                if self._progress_bar == ProgressBar.SOURCES:
                    progress_bar.set_postfix(
                        {"scenes": self._mp_scene_counter.value}, refresh=False
                    )
                    progress_bar.update(1)
                elif self._progress_bar == ProgressBar.SCENES:
                    progress_bar.set_postfix(
                        {"sources": self._mp_source_counter.value}, refresh=False
                    )
                    progress_bar.update(processed_scenes)

    @staticmethod
    def _optimal_chunksize(num_sources: int | None, num_processes: int | None) -> int:
        if num_sources is None:
            return 1

        num_processes = num_processes or mp.cpu_count()
        chunksize, extra = divmod(num_sources, num_processes * 4)
        chunksize += int(extra > 0)
        return max(chunksize, 1)

    @staticmethod
    def _process_fn(args: _ProcessArgs[T_ID, T_Source]) -> list[Scene[T_ID]]:
        """Worker process function that processes a single source and returns a list of Scenes."""
        loader, source = args.loader, args.source
        scenes: list[Scene[T_ID]] = []

        for scene_data, map_context in loader.process_next(source):
            scene = loader.create_scene(scene_data, source.identifier, map_context)
            scenes.append(scene)

        if num_scenes := len(scenes):
            with _scene_counter.get_lock():
                start_index = _scene_counter.value + 1
                _scene_counter.value += num_scenes

            for i in range(num_scenes):
                scenes[i] = replace(scenes[i], scene_number=start_index + i)

        with _source_counter.get_lock():
            _source_counter.value += 1

        return scenes

    @staticmethod
    def _process_fn_callback(args: _ProcessArgs[T_ID, T_Source]) -> int:
        """Worker process function that applies a callback to each Scene.

        This function returns the number of processed scenes (used for keeping track of progress),
        otherwise it behaves similarly to `_process_fn` but instead of returning the scenes, it
        applies the provided callable to each scene for side effects (e.g., saving to disk).
        """
        if args.fn is None:
            msg = "no callable function provided in _ProcessArgs for _process_fn_callable."
            raise ValueError(msg)

        processed_scenes = 0
        loader, source = args.loader, args.source
        scene_number: int = -1
        cb_args, cb_kwargs = args.cb_args or (), args.cb_kwargs or {}
        for scene_data, map_context in loader.process_next(source):
            # Should be fine to aquire lock in loop, since `process_next` is expected to be
            # relatively slow. Not as easy to move outside the loop here since we want to avoid
            # loading all scenes into memory at once, which would happen if we used `_process_fn`.
            with _scene_counter.get_lock():
                _scene_counter.value += 1
                scene_number = _scene_counter.value

            scene = loader.create_scene(scene_data, source.identifier, map_context)
            scene = replace(scene, scene_number=scene_number)
            args.fn(scene, *cb_args, **cb_kwargs)
            processed_scenes += 1

        with _source_counter.get_lock():
            _source_counter.value += 1

        return processed_scenes

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


class _ProcessArgs(NamedTuple, Generic[T_ID, T_Source]):
    """Arguments for processing a single source in a worker process."""

    source: Source[T_ID, T_Source]
    loader: BaseSceneLoader[T_ID, T_Source]
    fn: Callable[..., None] | None = None
    cb_args: tuple | None = None
    cb_kwargs: dict[str, Any] | None = None


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
