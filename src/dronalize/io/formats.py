"""Enumerations for persisted storage backends supported by Dronalize."""

from enum import Enum


class StorageBackend(str, Enum):
    """Supported persisted storage backends."""

    MDS = "mds"
    NULL = "null"
