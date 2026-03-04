"""Registry for dataset descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from dronalize.core.datatypes.split import DatasetSplit

if TYPE_CHECKING:
    from collections.abc import Callable

    from dronalize.core.datatypes.loader_config import LoaderConfig
    from dronalize.core.protocols.loader import BaseSceneLoader


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Everything needed to fully process a single dataset."""

    name: str
    """Canonical slug, e.g. "ind", "argoverse2", "waymo"."""

    loader_factory: Callable[..., BaseSceneLoader]
    """Factory function that creates a scene loader for the dataset."""

    default_config: LoaderConfig
    """Default loader configuration for the dataset."""

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
