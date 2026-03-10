"""Registry for dataset descriptors."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from enum import IntEnum, auto
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


class MapMode(IntEnum):
    """How a dataset exposes map data at runtime."""

    NONE = auto()
    BUILDER_ONLY = auto()
    INLINE = auto()
    LAZY_KEYED = auto()
    SHARED_SINGLE = auto()
    SHARED_KEYED = auto()


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

    map_mode: MapMode = MapMode.NONE
    """How this dataset exposes map data at runtime."""

    predefined_splits: list[DatasetSplit] | None = None
    """Predefined splits for the dataset, if any."""

    @property
    def has_map(self) -> bool:
        """Compatibility flag for code that only distinguishes map vs. no map."""
        return self.map_mode is not MapMode.NONE

    def with_splits(
        self, *splits: DatasetSplit | Literal["train", "val", "test"]
    ) -> DatasetDescriptor:
        """Return a copy of this descriptor with the specified predefined splits."""
        normalized_splits = [DatasetSplit(s) if isinstance(s, str) else s for s in splits]
        return replace(self, predefined_splits=normalized_splits)

    def with_all_splits(self) -> DatasetDescriptor:
        """Indicate that this dataset has all three standard splits (train, val, test)."""
        return self.with_splits(DatasetSplit.TEST, DatasetSplit.TRAIN, DatasetSplit.VAL)

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

    Raises
    ------
    ValueError
        If a dataset with the same canonical name has already been registered.
    """
    existing = _REGISTRY.get(descriptor.name)
    if existing is not None and existing != descriptor:
        msg = f"Dataset '{descriptor.name}' is already registered."
        raise ValueError(msg)
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

    Raises
    ------
    KeyError
        If the dataset name is unknown.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(available()) or "none"
        msg = f"Unknown dataset '{name}'. Available datasets: {known}."
        raise KeyError(msg) from exc


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
