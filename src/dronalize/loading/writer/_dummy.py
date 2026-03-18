from __future__ import annotations

import functools
from collections import Counter
from pprint import pprint
from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.loading import SceneWriter

if TYPE_CHECKING:
    from collections.abc import Callable

    from dronalize.categories import DatasetSplit
    from dronalize.scene import Scene


def create_writer(*, log: bool, identifier: int | None = None) -> DummyWriter:
    return DummyWriter(identifier, log=log)


class DummyWriter(SceneWriter):
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
            pprint(f"[{self._identifier}] Finished writing local.")
            pprint(f"[{self._identifier}] Scene counts: {dict(self._count)}")

    @override
    def finish_final(self) -> None:
        if self._log:
            pprint(f"[{self._identifier}] Finished writing final.")
