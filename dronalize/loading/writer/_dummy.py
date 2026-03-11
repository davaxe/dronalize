from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.loading import SceneWriter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from dronalize.categories import DatasetSplit
    from dronalize.scene import Scene


def create_writer(*, log: bool, identifier: int | None = None) -> DummyWriter:
    return DummyWriter(identifier, log=log)


class DummyWriter(SceneWriter):
    def __init__(self, identifier: str | int | None = None, *, log: bool = False) -> None:
        self._identifier: str = "UNNAMED" if identifier is None else str(identifier)
        self._log: bool = log

    @classmethod
    @override
    def as_factory(
        cls,
        *,
        log: bool = False,
    ) -> Callable[[int | None], DummyWriter]:
        return functools.partial(cls, log=log)

    @override
    def set_output_dir(self, output_dir: Path) -> None: ...

    @override
    def write(
        self,
        processed: Scene,
        splits: Iterator[DatasetSplit] | None = None,
        *,
        strict: bool = False,
    ) -> bool:
        return True

    @override
    def finish_local(self) -> None:
        if self._log:
            print(f"[{self._identifier}] Finished writing local.")

    @override
    def finish_final(self) -> None:
        if self._log:
            print(f"[{self._identifier}] Finished writing final.")
