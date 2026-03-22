from enum import StrEnum


class OutputFormat(StrEnum):
    """Supported persisted export formats."""

    MDS = "mds"
    ZARR = "zarr"
    DUMMY = "dummy"
