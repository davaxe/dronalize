"""Enumerations and helpers for persisted storage backends."""

from enum import Enum


class StorageBackend(str, Enum):
    """Supported persisted storage backends."""

    MDS = "mds"
    NULL = "null"


def parse_storage_backend(value: StorageBackend | str) -> StorageBackend:
    """Normalize a storage-backend identifier into the canonical enum value."""
    return value if isinstance(value, StorageBackend) else StorageBackend(value)
