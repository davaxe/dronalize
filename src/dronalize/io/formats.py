"""Enumerations for persisted output formats supported by Dronalize."""

from enum import Enum


class OutputFormat(str, Enum):
    """Supported persisted export formats."""

    MDS = "mds"
    DUMMY = "dummy"
