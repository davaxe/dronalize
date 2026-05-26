from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, runtime_checkable

from typing_extensions import Self, TypeVar

from dronalize.core.categories import DatasetSplit
from dronalize.core.scene import Scene
from dronalize.io.records import SceneRecord

if TYPE_CHECKING:
    from collections.abc import Iterator


SampleT = TypeVar("SampleT", default=SceneRecord)


RecordTransform: TypeAlias = Callable[[SceneRecord], SampleT]
"""Callable that converts a canonical `SceneRecord` into a persisted sample.

This is the preferred customization hook for user-defined persisted sample
layouts because it preserves Dronalize's standard output semantics before
materializing the custom payload.
"""


SceneTransform: TypeAlias = Callable[[Scene], SampleT]
"""Callable that converts a runtime `Scene` directly into a persisted sample.

This is an advanced escape hatch for users who intentionally want to bypass
`SceneRecord` encoding. Callers using this hook own schema conversion,
dtype policy, recentering, map resolution, and reader compatibility.
"""


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


def validate_transform_choice(
    *, record_transform: object | None, scene_transform: object | None
) -> None:
    """Validate that at most one sample customization hook is configured."""
    if record_transform is not None and scene_transform is not None:
        msg = "Use either `record_transform` or `scene_transform`, not both."
        raise ValueError(msg)
