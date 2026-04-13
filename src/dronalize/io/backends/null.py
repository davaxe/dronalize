"""No-op writer backend.

The ``null`` backend executes the full runtime, counts accepted scenes, and
intentionally skips all persisted scene output. It is useful for tests, dry
runs, configuration validation, and measuring pipeline cost without storage
I/O.
"""

from __future__ import annotations

import functools
import multiprocessing as mp
from collections import Counter
from typing import TYPE_CHECKING, final

from typing_extensions import override

from dronalize.core.categories import DatasetSplit
from dronalize.io.base import DatasetWriter

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.sharedctypes import Synchronized

    from dronalize.core.scene import Scene


@final
class NullWriter(DatasetWriter):
    """No-op writer used by tests and dry-run execution paths."""

    _counts_shared: dict[str, Synchronized[int]] | None = None

    def __init__(
        self,
        identifier: str | int | None = None,
        *,
        log: bool = False,
        count_shared: dict[str, Synchronized[int]] | None = None,
    ) -> None:
        self._identifier: str = "UNNAMED" if identifier is None else str(identifier)
        self._log: bool = log
        self._count: Counter[str] = Counter()
        self._count_shared = count_shared

    @classmethod
    @override
    def as_factory(cls, *, log: bool = False) -> Callable[[int | None], NullWriter]:
        """Create a factory that shares split counters across workers."""
        if cls._counts_shared is None:
            cls._counts_shared = {
                "unsplit": mp.Value("i", 0),
                DatasetSplit.TRAIN.value: mp.Value("i", 0),
                DatasetSplit.VAL.value: mp.Value("i", 0),
                DatasetSplit.TEST.value: mp.Value("i", 0),
            }

        return functools.partial(cls, log=log, count_shared=cls._counts_shared)

    @override
    def write(self, scene: Scene) -> None:
        """Count one scene without writing any output."""
        effective_split = scene.split_assignment
        self._count["scenes"] += 1
        self._count[effective_split.value if effective_split else "unsplit"] += 1

    @override
    def finish_local(self) -> None:
        """Merge this worker's local counters into the shared totals."""
        for split, count in self._count.items():
            if self._count_shared is not None and split in self._count_shared:
                with self._count_shared[split].get_lock():
                    self._count_shared[split].value += count

    @override
    def finish_final(self) -> None:
        """Optionally print the aggregated split counts for the full run."""
        if self._log:
            if self._count_shared is None:
                return

            total = sum(v.value for v in self._count_shared.values())
            split_counts: dict[str, int] = {
                split: v.value for split, v in self._count_shared.items()
            }
            print(f"Total: {total:,} scenes")
            for split, count in split_counts.items():
                pct = (count / total * 100) if total > 0 else 0.0
                print(f"  {split:<12} {count:>10,}  ({pct:6.2f}%)")

            for k in self._count_shared or {}:
                with self._count_shared[k].get_lock():
                    self._count_shared[k].value = 0
