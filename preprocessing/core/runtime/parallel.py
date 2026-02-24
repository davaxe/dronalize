from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Hashable, Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

from typing_extensions import Self, override

from preprocessing.core.interface.trajectory import DataProcessor, ProcessorConfig, Scene

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

    import polars as pl

T_Source = TypeVar("T_Source")
T_ID = TypeVar("T_ID", bound=(Hashable))


class ParallelDataProcessor(DataProcessor[T_ID, T_Source]):
    """A generic parallel data processor that wraps another Dataprocessor.

    It uses Python's multiprocessing module to parallelize the processing of sources across multiple
    CPU cores. The number of processes and chunksize can be configured depending on workload.

    By default the order resulting scenes when using `scenes_iter` method will not be deterministic
    across runs or systems due to the use of `imap_unordered` for better performance. If a
    deterministic order is required, the `maintain_order` flag can be set to True, which will use `imap`
    instead of `imap_unordered`.

    """

    def __init__(
        self,
        processor: DataProcessor[T_ID, T_Source],
        chunksize: int = 1,
        processes: int | None = None,
        *,
        maintain_order: bool = False,
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

        """
        self._processor = processor
        self._global_counter: Synchronized[int] = mp.Value("i", 0)
        self._chunksize: int = chunksize
        self._processes: int | None = processes
        self._maintain_order: bool = maintain_order

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
    def default_config(self) -> ProcessorConfig:
        return self._processor.default_config()

    @override
    def scenes_iter(self) -> Iterable[Scene[T_ID]]:
        if self._processes == 1:
            # Faster to avoid multiprocessing overhead if only using one process.
            yield from super().scenes_iter()
            return

        with mp.Pool(
            self._processes, initializer=_init_worker, initargs=(self._global_counter,)
        ) as pool:
            map_fn = pool.imap if self._maintain_order else pool.imap_unordered
            for scene in map_fn(
                ParallelDataProcessor._process_fn,
                (_ProcessArgs(s, self._processor) for s in self._processor.sources()),
                self._chunksize,
            ):
                yield from scene

    @staticmethod
    def _process_fn(args: _ProcessArgs[T_ID, T_Source]) -> list[Scene[T_ID]]:
        processor = args.processor
        scenes: list[Scene[T_ID]] = []
        scene_number: int = -1
        for scene_data in processor.process_next(args.source[1]):
            with _worker_counter.get_lock():
                _worker_counter.value += 1
                scene_number = _worker_counter.value

            scene = processor.create_scene(scene_data, args.source[0])
            scenes.append(replace(scene, scene_number=scene_number))

        return scenes


class _ProcessArgs(NamedTuple, Generic[T_ID, T_Source]):
    """Arguments for processing a single source in a worker process."""

    source: tuple[T_ID, T_Source]
    processor: DataProcessor[T_ID, T_Source]


# Worker initialization function to set up the global counter in each worker process.
_worker_counter: Synchronized[int]


def _init_worker(counter: Synchronized[int]) -> None:
    # This is the standard and most efficent way to share a counter across processes in Python's
    # multiprocessing module.
    global _worker_counter  # noqa: PLW0603
    _worker_counter = counter


if __name__ == "__main__":
    from pathlib import Path

    from preprocessing.datasets.pedestrian.trajectory_processor import EthUcyProcessor

    processor_single = EthUcyProcessor(data_root=Path("data"), dataset="hotel", split="train")
    processor = ParallelDataProcessor(processor_single, maintain_order=False)
    start_time = time.perf_counter()
    for _scene in processor.scenes_iter():
        continue
    multi_time = time.perf_counter() - start_time
    print(
        f"Processed all scenes ({processor._global_counter.value}) in {multi_time:.2f} seconds with multiprocessing.",
    )

    start_time = time.perf_counter()
    for _scene in processor_single.scenes_iter():
        continue

    single_time = time.perf_counter() - start_time
    print(
        f"Processed all scenes ({processor_single._count}) in {single_time:.2f} seconds without multiprocessing.",
    )

    print(f"Multiprocessing speedup: {single_time / multi_time:.2f}x")
