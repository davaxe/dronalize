from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.categories import DatasetSplit


class DronalizeError(Exception):
    """Base exception class for all dronalize errors."""


class LoaderConfigError(DronalizeError, ValueError):
    """Raised when there is an issue with the loader configuration."""


class SplitNotSupportedError(ValueError):
    """Raised when a dataset does not support the requested split.

    This error is raised when a specific split (e.g., `TRAIN`, `VAL`,
    `TEST`) is requested from a dataset that does not provide predefined
    splits.

    Parameters
    ----------
    loader_name : str
        The name of the loader class that does not support splits.
    split : DatasetSplit
        The split that was requested.

    """

    def __init__(self, loader_name: str, split: DatasetSplit) -> None:
        super().__init__(f"{loader_name} does not support split '{split}'. ")
        self.loader_name: str = loader_name
        self.split: DatasetSplit = split
