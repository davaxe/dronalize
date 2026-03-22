from __future__ import annotations

import functools
from collections import Counter
from typing import TYPE_CHECKING, final

from typing_extensions import override

from dronalize.storage.spec import FORMAT_VERSION, StorageManifest
from dronalize.storage.writers.protocol import SceneWriter

if TYPE_CHECKING:
    from collections.abc import Callable

    from dronalize.categories import DatasetSplit
    from dronalize.scene import Scene


def create_writer(*, log: bool, identifier: int | None = None) -> DummyWriter:
    return DummyWriter(identifier, log=log)


@final
class DummyWriter(SceneWriter):
    """No-op writer used by tests and dry-run execution paths."""

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
    )

    def __init__(self, identifier: str | int | None = None, *, log: bool = False) -> None:
        self._identifier: str = "UNNAMED" if identifier is None else str(identifier)
        self._log: bool = log
        self._count: Counter[str] = Counter()

    @classmethod
    @override
    def as_factory(
        cls,
        *,
        log: bool = False,
    ) -> Callable[[int | None], DummyWriter]:
        return functools.partial(cls, log=log)

    @override
    def write(
        self,
        scene: Scene,
        split: DatasetSplit | None = None,
    ) -> bool:
        effective_split = split if split is not None else scene.split_assignment
        self._count["scenes"] += 1
        self._count[effective_split.value if effective_split else "unsplit"] += 1
        return True

    @override
    def finish_local(self) -> None:
        if self._log:
            pass

    @override
    def finish_final(self) -> None:
        if self._log:
            pass
