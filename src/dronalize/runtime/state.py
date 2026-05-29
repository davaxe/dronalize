# ruff: noqa: D102
"""Shared multiprocessing state used by the internal parallel runner."""

from __future__ import annotations

import multiprocessing as mp
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

from typing_extensions import TypedDict

from dronalize.core.categories import DatasetSplit

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext
    from multiprocessing.sharedctypes import Synchronized
    from multiprocessing.synchronize import Event, Lock

    from dronalize.io.base import DatasetWriter
    from dronalize.runtime.processor import RuntimeProcessor


class SplitCounts(TypedDict, total=True):
    unsplit: int
    train: int
    val: int
    test: int


@dataclass(frozen=True, slots=True)
class Progress:
    running: bool
    """Whether the execution is currently running."""
    processed_sources: int
    """The number of sources that have been processed."""
    candidate_scenes: int
    """The number of candidate scenes that have been generated and screened.

    This is incremented for every scene that is generated and screened,
    regardless of whether it is selected or not.
    """
    selected_scenes: int
    """Actual number of scenes that have been selected for the dataset."""
    total_sources: int | None
    """Total sources to process if known, otherwise None."""
    scene_limit: int | None
    """The total scene limit if one is set, otherwise None."""
    active_workers: int
    """Current number of active worker processes."""
    split_counts: SplitCounts
    """Split partition counts, with keys "unsplit", "train", "val", and "test"."""
    screening_enabled: bool
    """Whether the processor has screening enabled."""


@dataclass(slots=True)
class WorkerRegistry:
    next_worker_id: Synchronized[int]

    @classmethod
    def create(cls, mp_context: BaseContext | None = None) -> WorkerRegistry:
        ctx = mp_context or mp.get_context()
        return cls(next_worker_id=ctx.Value("i", 0))

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
    candidate_scene_counter: Synchronized[int]
    selected_scene_counter: Synchronized[int]
    source_counter: Synchronized[int]
    unsplit_counter: Synchronized[int]
    train_counter: Synchronized[int]
    val_counter: Synchronized[int]
    test_counter: Synchronized[int]
    snapshot_lock: Lock
    update_event: Event

    @classmethod
    def create(cls, mp_context: BaseContext | None = None) -> ProgressState:
        ctx = mp_context or mp.get_context()
        return cls(
            active_workers=ctx.Value("i", 0),
            candidate_scene_counter=ctx.Value("i", 0),
            selected_scene_counter=ctx.Value("i", 0),
            source_counter=ctx.Value("i", 0),
            unsplit_counter=ctx.Value("i", 0),
            train_counter=ctx.Value("i", 0),
            val_counter=ctx.Value("i", 0),
            test_counter=ctx.Value("i", 0),
            snapshot_lock=ctx.Lock(),
            update_event=ctx.Event(),
        )

    def reset(self) -> None:
        counters = (
            self.active_workers,
            self.candidate_scene_counter,
            self.selected_scene_counter,
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

    def record_candidate_scene(self) -> int:
        with self.candidate_scene_counter.get_lock():
            self.candidate_scene_counter.value += 1
            value = self.candidate_scene_counter.value
        self.update_event.set()
        return value

    def claim_selected_scene(self, limit: int | None = None) -> int | None:
        with self.selected_scene_counter.get_lock():
            if limit is not None and self.selected_scene_counter.value >= limit:
                return None
            scene_number = self.selected_scene_counter.value
            self.selected_scene_counter.value += 1
        self.update_event.set()
        return scene_number

    def selected_scene_limit_reached(self, limit: int | None = None) -> bool:
        if limit is None:
            return False
        with self.selected_scene_counter.get_lock():
            return self.selected_scene_counter.value >= limit

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

    def split_counts(self) -> SplitCounts:
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


@dataclass(slots=True)
class SharedResources:
    registry: WorkerRegistry
    progress: ProgressState
    scene_limit: int | None = None

    @classmethod
    def create(
        cls, *, scene_limit: int | None = None, mp_context: BaseContext | None = None
    ) -> SharedResources:
        return cls(
            registry=WorkerRegistry.create(mp_context),
            progress=ProgressState.create(mp_context),
            scene_limit=scene_limit,
        )

    def reset(self) -> None:
        self.registry.reset()
        self.progress.reset()


@dataclass(slots=True)
class WorkerRuntime:
    shared: SharedResources
    worker_id: int
    processor: RuntimeProcessor | None = None
    writer: DatasetWriter | None = None
