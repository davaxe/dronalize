"""Enumerations and helpers for persisted storage backends."""

from enum import Enum


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
