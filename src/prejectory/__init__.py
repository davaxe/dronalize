# ruff: file-ignore[non-empty-init-module]
"""Trajectory processing: plan a request, run it, and open the resulting dataset.

Specialized models live in `config`, `core`, `datasets`, `runtime` and
`io`. Optional frameworks are imported only when their adapters are used.
"""

from importlib.metadata import PackageNotFoundError, version

from prejectory.io import open_dataset
from prejectory.runtime import ExecutionRequest, plan, run

__all__ = ["ExecutionRequest", "open_dataset", "plan", "run"]

try:
    __version__ = version("prejectory")
except PackageNotFoundError:
    __version__ = "0+unknown"
