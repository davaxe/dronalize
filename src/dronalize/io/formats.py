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


StorageBackendId = StorageBackend | str
"""Storage backend identifier accepted by public runtime APIs."""


def storage_backend_name(backend: StorageBackendId) -> str:
    """Return the stable string key for a storage backend identifier."""
    return backend.value if isinstance(backend, StorageBackend) else str(backend)
