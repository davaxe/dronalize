"""Dataset split definitions for loading specific subsets of data."""

from __future__ import annotations

from enum import StrEnum


class DatasetSplit(StrEnum):
    """Enum representing the available dataset splits."""

    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    ALL = "all"


class SplitNotSupportedError(ValueError):
    """Raised when a dataset does not support the requested split.

    This error is raised when a specific split (e.g., `TRAIN`, `TEST`,
    `VALIDATE`) is requested from a dataset that does not provide predefined
    splits. In such cases, only `DatasetSplit.ALL` is valid.

    Parameters
    ----------
    loader_name : str
        The name of the loader class that does not support splits.
    split : DatasetSplit
        The split that was requested.

    """

    def __init__(self, loader_name: str, split: DatasetSplit) -> None:
        super().__init__(
            f"{loader_name} does not support split '{split.value}'. "
            f"This dataset has no predefined splits. "
            f"Use DatasetSplit.ALL (or omit the split parameter) to load all data."
        )
        self.loader_name = loader_name
        self.split = split
