from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Callable

    from dronalize.categories import DatasetSplit
    from dronalize.scene import Scene
    from dronalize.storage.spec import StorageManifest


@runtime_checkable
class StorageWriter(Protocol):
    """Protocol for writing processed scenes to persisted storage."""

    manifest: StorageManifest

    def write(
        self,
        scene: Scene,
        split: DatasetSplit | None = None,
    ) -> bool:
        """Write a single processed scene."""
        ...

    def finish_local(self) -> None:
        """Finalize the current worker or process."""
        ...

    def finish_final(self) -> None:
        """Finalize the full dataset once all workers are done."""
        ...

    @classmethod
    def as_factory(cls, *args: Any, **kwargs: Any) -> Callable[[int | None], Self]:  # noqa: ANN401
        """Create a worker-local writer factory."""
        return functools.partial(cls, *args, **kwargs)


SceneWriter = StorageWriter
