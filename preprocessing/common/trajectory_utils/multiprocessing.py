from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Hashable, Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

from typing_extensions import override

from preprocessing.core.interface.trajectory import DataProcessor, ProcessorConfig, Scene

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

    import polars as pl

T_Source = TypeVar("T_Source")
T_ID = TypeVar("T_ID", bound=(Hashable))

_worker_counter: Synchronized[int]


class _ProcessArgs(NamedTuple, Generic[T_ID, T_Source]):
    source: tuple[T_ID, T_Source]
    processor: DataProcessor[T_ID, T_Source]


class MultiProcessingDataProcessor(DataProcessor[T_ID, T_Source], Generic[T_ID, T_Source]):
    def __init__(
        self,
        processor: DataProcessor[T_ID, T_Source],
    ) -> None:
        self._processor = processor
        self._global_counter: Synchronized[int] = mp.Value("i", 0)

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
    def scenes_iter(
        self, processes: int | None = None, chunksize: int = 1
    ) -> Iterable[Scene[T_ID]]:
        with mp.Pool(processes, initializer=_init_worker, initargs=(self._global_counter,)) as pool:
            for scene in pool.imap_unordered(
                MultiProcessingDataProcessor._process_fn,
                (_ProcessArgs(s, self._processor) for s in self._processor.sources()),
                chunksize,
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


def _init_worker(counter: Synchronized[int]) -> None:
    # This is the standard and most efficent way to share a counter across processes in Python's
    # multiprocessing module.
    global _worker_counter  # noqa: PLW0603
    _worker_counter = counter


if __name__ == "__main__":
    from pathlib import Path

    from preprocessing.datasets.pedestrian.trajectory_processor import EthUcyProcessor

    processor_single = EthUcyProcessor(data_root=Path("data"), dataset="hotel", split="train")
    processor = MultiProcessingDataProcessor(processor_single)
    start_time = time.perf_counter()
    for _scene in processor.scenes_iter(None, 1):
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
