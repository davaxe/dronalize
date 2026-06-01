from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Generic, Protocol, TypeAlias, runtime_checkable

from typing_extensions import TypeVar, override

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


class StorageBackend(str, Enum):
    """Supported persisted storage backends."""

    MDS = "mds"
    """MDS storage backend.

    ??? warning "Extra dependencies"
        Using MDS requires installing the `dronalize[mds]` extra:

        ```sh
        pip install dronalize[mds]
        ```
    """
    PICKLE = "pickle"
    """Pickle storage backend. Requires no extra dependencies."""
    NULL = "null"
    """Null storage backend that discards all data. Useful for testing."""


StorageBackendId = StorageBackend | str
"""Storage backend identifier accepted by public runtime APIs."""


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

    def get(self, at: int) -> SampleT | None:
        """Return a single decoded scene record.

        Parameters
        ----------
        at : int
            Index of the scene record to return.

        Returns
        -------
        SampleT or None
            The decoded scene record at the specified index, or `None` if the
            index is out of bounds.

        """
        try:
            return self[at]
        except IndexError:
            return None


class IterableDatasetReader(ABC, Generic[SampleT]):
    """Abstract base class for iterable scene readers.

    This is a specialization of `DatasetReader` for backends that support
    streaming data without random access.

    """

    @abstractmethod
    def __iter__(self) -> Iterator[SampleT]:
        """Iterate over decoded scene records."""

    def __len__(self) -> int:
        """Return the number of scene records, if known."""
        msg = f"{self.__class__.__name__} does not implement __len__"
        raise NotImplementedError(msg)


@runtime_checkable
class DatasetWriter(Protocol):
    """Protocol for writing processed scenes to persisted storage."""

    def write(self, scene: Scene) -> None:
        """Write one processed scene and return whether it was accepted."""
        ...

    def finish_local(self) -> None:
        """Finalize worker-local state once the current worker is done."""
        _ = self


class WriterProvider(ABC):
    """Provider that owns writer lifecycle for one execution run."""

    @abstractmethod
    def open_worker(self, worker_id: int) -> DatasetWriter:
        """Create the writer used by one worker."""

    def finish_final(self) -> None:
        """Finalize dataset-wide writer state after all workers finish."""
        _ = self


def _noop_finish_final() -> None:
    return


@dataclass(frozen=True, slots=True)
class WorkerWriterProvider(WriterProvider):
    """Pickleable writer provider backed by top-level callables."""

    create_worker: Callable[[int], DatasetWriter]
    finalize: Callable[[], None] = _noop_finish_final

    @override
    def open_worker(self, worker_id: int) -> DatasetWriter:
        """Create the writer used by one worker."""
        return self.create_worker(worker_id)

    @override
    def finish_final(self) -> None:
        """Finalize dataset-wide writer state after all workers finish."""
        self.finalize()


def split_directory_name(split: DatasetSplit | str | None) -> str:
    """Return the storage subdirectory name for one dataset split."""
    if split is None:
        return "unsplit"
    return split.value if isinstance(split, DatasetSplit) else str(split)


def validate_transform_choice(
    *,
    record_transform: RecordTransform[object] | None,
    scene_transform: SceneTransform[object] | None,
) -> None:
    """Validate that at most one sample customization hook is configured."""
    if record_transform is not None and scene_transform is not None:
        msg = "Use either `record_transform` or `scene_transform`, not both."
        raise ValueError(msg)


def storage_backend_name(backend: StorageBackendId) -> str:
    """Return the stable string key for a storage backend identifier."""
    return backend.value if isinstance(backend, StorageBackend) else str(backend)
