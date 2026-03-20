from enum import StrEnum


class OutputFormat(StrEnum):
    """Enum for output formats."""

    MDS = "mds"
    ZARR = "zarr"
    DUMMY = "dummy"
