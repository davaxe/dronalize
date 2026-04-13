from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Protocol, runtime_checkable

from typing_extensions import Self, TypeVar

from dronalize.core.categories import DatasetSplit
from dronalize.io.records import SceneRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from dronalize.core.scene import Scene

SampleT = TypeVar("SampleT", default=SceneRecord)


class DatasetReader(ABC, Generic[SampleT]):
    """Abstract base class for scene readers."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of scene records."""

    def __iter__(self) -> Iterator[SampleT]:
        """Iterate over decoded scene records."""
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def __getitem__(self, at: int) -> SampleT:
        """Return a single decoded scene record."""


@runtime_checkable
class DatasetWriter(Protocol):
    """Protocol for writing processed scenes to persisted storage."""

    def write(self, scene: Scene) -> None:
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


def split_directory_name(split: DatasetSplit | str | None) -> str:
    """Return the storage subdirectory name for one dataset split."""
    if split is None:
        return "unsplit"
    return split.value if isinstance(split, DatasetSplit) else str(split)
