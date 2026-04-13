"""Shared multiprocessing state used by the internal parallel runner."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dronalize.core.categories import DatasetSplit

if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing.sharedctypes import Synchronized
    from multiprocessing.synchronize import Event, Lock

    from dronalize.io.base import DatasetWriter
    from dronalize.processing.loading.base import BaseSceneLoader
    from dronalize.runtime._internal.scene import SceneBuilder


@dataclass(frozen=True, slots=True)
class Progress:
    running: bool
    processed_sources: int
    processed_scenes: int
    total_sources: int | None
    total_scenes: int | None
    active_workers: int
    split_counts: dict[str, int]


@dataclass(slots=True)
class WorkerRegistry:
    next_worker_id: Synchronized[int]

    @classmethod
    def create(cls) -> WorkerRegistry:
        return cls(next_worker_id=mp.Value("i", 0))

    def reset(self) -> None:
        with self.next_worker_id.get_lock():
            self.next_worker_id.value = 0

    def next_worker(self) -> int:
        with self.next_worker_id.get_lock():
            worker_id = self.next_worker_id.value
            self.next_worker_id.value += 1
            return worker_id


@dataclass(slots=True)
class ProgressState:
    active_workers: Synchronized[int]
    scene_counter: Synchronized[int]
    source_counter: Synchronized[int]
    unsplit_counter: Synchronized[int]
    train_counter: Synchronized[int]
    val_counter: Synchronized[int]
    test_counter: Synchronized[int]
    snapshot_lock: Lock
    update_event: Event

    @classmethod
    def create(cls) -> ProgressState:
        return cls(
            active_workers=mp.Value("i", 0),
            scene_counter=mp.Value("i", 0),
            source_counter=mp.Value("i", 0),
            unsplit_counter=mp.Value("i", 0),
            train_counter=mp.Value("i", 0),
            val_counter=mp.Value("i", 0),
            test_counter=mp.Value("i", 0),
            snapshot_lock=mp.Lock(),
            update_event=mp.Event(),
        )

    def reset(self) -> None:
        counters = (
            self.active_workers,
            self.scene_counter,
            self.source_counter,
            self.unsplit_counter,
            self.train_counter,
            self.val_counter,
            self.test_counter,
        )
        with ExitStack() as stack:
            for counter in counters:
                _ = stack.enter_context(counter.get_lock())
            for counter in counters:
                counter.value = 0
            _ = self.update_event.clear()

    def claim_scene(self, limit: int | None = None) -> int | None:
        with self.scene_counter.get_lock():
            if limit is not None and self.scene_counter.value >= limit:
                return None
            scene_number = self.scene_counter.value
            self.scene_counter.value += 1
        self.update_event.set()
        return scene_number

    def increment_source(self) -> int:
        with self.source_counter.get_lock():
            self.source_counter.value += 1
            value = self.source_counter.value
        self.update_event.set()
        return value

    def record_split(self, split: DatasetSplit | None) -> None:
        counter = {
            None: self.unsplit_counter,
            DatasetSplit.TRAIN: self.train_counter,
            DatasetSplit.VAL: self.val_counter,
            DatasetSplit.TEST: self.test_counter,
        }[split]
        with counter.get_lock():
            counter.value += 1
        self.update_event.set()

    def split_counts(self) -> dict[str, int]:
        return {
            "unsplit": self.unsplit_counter.value,
            DatasetSplit.TRAIN.value: self.train_counter.value,
            DatasetSplit.VAL.value: self.val_counter.value,
            DatasetSplit.TEST.value: self.test_counter.value,
        }

    def worker_started(self) -> None:
        with self.active_workers.get_lock():
            self.active_workers.value += 1
        self.update_event.set()

    def worker_stopped(self) -> int:
        with self.active_workers.get_lock():
            self.active_workers.value -= 1
            value = self.active_workers.value
        self.update_event.set()
        return value

    def active_worker_count(self) -> int:
        with self.active_workers.get_lock():
            return self.active_workers.value


@dataclass(slots=True)
class SharedResources:
    registry: WorkerRegistry
    progress: ProgressState
    scene_limit: int | None = None

    @classmethod
    def create(cls, *, scene_limit: int | None = None) -> SharedResources:
        return cls(
            registry=WorkerRegistry.create(),
            progress=ProgressState.create(),
            scene_limit=scene_limit,
        )

    def reset(self) -> None:
        self.registry.reset()
        self.progress.reset()


@dataclass(slots=True)
class WorkerRuntime:
    shared: SharedResources
    worker_id: int
    loader: BaseSceneLoader[Any, Any] | None = None
    builder: SceneBuilder | None = None
    writer: DatasetWriter | None = None

    def active_workers(self) -> int:
        return self.shared.progress.active_worker_count()

    @contextlib.contextmanager
    def progress_snapshot_lock(self) -> Generator[None, None, None]:
        with self.shared.progress.snapshot_lock:
            yield
