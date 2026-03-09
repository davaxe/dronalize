"""Registry for dataset descriptors."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from dronalize.core.datatypes.map_config import MapConfig
from dronalize.core.datatypes.split import DatasetSplit

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from dronalize.core.datatypes.loader_config import LoaderConfig
    from dronalize.core.protocols.loader import BaseSceneLoader


class DatasetLifecycleContext(Protocol):
    """Defines the resource lifecycle contract for a dataset.

    Implementations must act as context managers that perform necessary
    initialization (e.g., allocating shared memory for maps) before yielding,
    and guarantee resource cleanup after the context exits.

    """

    def __call__(
        self, root: Path, loader_config: LoaderConfig, map_config: MapConfig
    ) -> AbstractContextManager[None]:
        """Execute the dataset lifecycle context.

        Parameters
        ----------
        root : Path
            Root directory of the dataset.
        loader_config : LoaderConfig
            Configuration for the dataset loader.
        map_config : MapConfig
            Configuration for building or loading the map graph.

        Returns
        -------
        AbstractContextManager[None]
            A context manager governing the dataset's temporary resources.

        """
        ...


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Everything needed to fully process a single dataset."""

    name: str
    """Canonical slug, e.g. "ind", "argoverse2", "waymo"."""

    loader_factory: Callable[..., BaseSceneLoader]
    """Factory function that creates a scene loader for the dataset."""

    default_config: LoaderConfig
    """Default loader configuration for the dataset."""

    default_map_config: MapConfig = field(default_factory=MapConfig.default)
    """Default map configuration for the dataset, if applicable."""

    lifecycle_context: DatasetLifecycleContext | None = None
    """Optional lifecycle context for the dataset, which can manage resources like shared memory."""

    has_map: bool = False
    """Whether the dataset has available map data."""

    predefined_splits: list[DatasetSplit] | None = None
    """Predefined splits for the dataset, if any."""

    def with_splits(
        self, splits: list[DatasetSplit | Literal["test", "val", "train"]] | None
    ) -> DatasetDescriptor:
        """Return a copy of this descriptor with the specified predefined splits."""
        return DatasetDescriptor(
            name=self.name,
            loader_factory=self.loader_factory,
            has_map=self.has_map,
            default_config=self.default_config,
            predefined_splits=[DatasetSplit(s) if isinstance(s, str) else s for s in splits]
            if splits is not None
            else None,
        )

    def with_all_splits(self) -> DatasetDescriptor:
        """Indicate that this dataset has all three standard splits (train, val, test)."""
        return self.with_splits([DatasetSplit.TEST, DatasetSplit.TRAIN, DatasetSplit.VAL])

    @contextmanager
    def execute_lifecycle_context(
        self, root: Path, loader_config: LoaderConfig, map_config: MapConfig
    ) -> Generator[None, None, None]:
        """Execute the lifecycle context manager, if defined."""
        if self.lifecycle_context is not None:
            with self.lifecycle_context(root, loader_config, map_config):
                yield
        else:
            yield


_REGISTRY: dict[str, DatasetDescriptor] = {}


def register(descriptor: DatasetDescriptor) -> DatasetDescriptor:
    """Register a dataset descriptor.

    Parameters
    ----------
    descriptor : DatasetDescriptor
        The descriptor to register.

    """
    _REGISTRY[descriptor.name] = descriptor
    return descriptor


def get(name: str) -> DatasetDescriptor:
    """Get a dataset descriptor by name.

    Parameters
    ----------
    name : str
        The name of the dataset to retrieve.

    Returns
    -------
    DatasetDescriptor
        The descriptor for the requested dataset.

    """
    return _REGISTRY[name]


def available() -> list[str]:
    """Get a list of available dataset names.

    Returns
    -------
    list[str]
        A sorted list of registered dataset names.
    """
    return sorted(_REGISTRY.keys())


if __name__ == "__main__":
    from dronalize.datasets import available as _av

    print("Available datasets:", _av())
