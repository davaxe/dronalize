from enum import Enum


class OutputFormat(str, Enum):
    """Enum for output formats."""

    MDS = "mds"
    ZARR = "zarr"
    DUMMY = "dummy"
