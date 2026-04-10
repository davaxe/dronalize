"""Protocols shared by persisted scene-writer backends."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Callable

    from dronalize.core.scene import Scene


@runtime_checkable
class DatasetWriter(Protocol):
    """Protocol for writing processed scenes to persisted storage."""

    def write(self, scene: Scene) -> bool:
        """Write one processed scene and return whether it was accepted."""
        ...

    def finish_local(self) -> None:
        """Finalize worker-local state once the current worker is done."""
        ...

    def finish_final(self) -> None:
        """Finalize dataset-wide state once all workers are done."""
        ...

    @classmethod
    def as_factory(cls, *args: Any, **kwargs: Any) -> Callable[[int | None], Self]:  # noqa: ANN401
        """Create a worker-local writer factory for the configured backend."""
        return functools.partial(cls, *args, **kwargs)
