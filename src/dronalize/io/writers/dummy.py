"""No-op writer backend used for tests and dry-run style execution."""

from __future__ import annotations

import functools
import multiprocessing as mp
from collections import Counter
from typing import TYPE_CHECKING, final

from typing_extensions import override

from dronalize.core.categories import DatasetSplit
from dronalize.io.manifest import FORMAT_VERSION, StorageManifest
from dronalize.io.writers.base import SceneWriter

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.sharedctypes import Synchronized

    from dronalize.core.scene import Scene


def create_writer(*, log: bool, identifier: int | None = None) -> DummyWriter:
    """Construct a ``DummyWriter`` instance directly.

    Parameters
    ----------
    log : bool
        Whether ``finish_final()`` should print aggregated counts.
    identifier : int | None, optional
        Optional identifier attached to the writer instance.

    Returns
    -------
    DummyWriter
        Configured dummy writer instance.
    """
    return DummyWriter(identifier, log=log)


@final
class DummyWriter(SceneWriter):
    """No-op writer used by tests and dry-run execution paths."""

    _counts_shared: dict[str, Synchronized[int]] | None = None

    manifest: StorageManifest = StorageManifest(
        format_version=FORMAT_VERSION,
        source_scene_schema="dummy",
        scene_schema="dummy",
        derived_features=(),
        feature_columns=(),
        input_len=0,
        output_len=0,
        precision="float32",
        offset_positions=False,
        has_map=False,
        sample_time=1.0,
        original_sample_time=1.0,
    )

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
    def as_factory(cls, *, log: bool = False) -> Callable[[int | None], DummyWriter]:
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
    def write(self, scene: Scene, split: DatasetSplit | None = None) -> bool:
        """Count one scene without writing any output."""
        effective_split = split if split is not None else scene.split_assignment
        self._count["scenes"] += 1
        self._count[effective_split.value if effective_split else "unsplit"] += 1
        return True

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
