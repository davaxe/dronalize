# ruff: file-ignore[non-empty-init-module]
"""Top-level package namespace for the `prejectory` library.

This module is intentionally small. It does not provide convenience aliases
for the main Python API. Instead, the public surface is organized around
explicit package namespaces so imports stay predictable and module ownership
remains clear.

Use the package namespaces directly:

- [`prejectory.datasets`][] for dataset lookup, descriptors, and registration
- [`prejectory.runtime`][] for config resolution, planning, and run-state models
- [`prejectory.processing`][] for processing config and grouped processing modules
- [`prejectory.io`][] for export config, manifests, readers, and adapters
- [`prejectory.core.scene`][] and [`prejectory.core.maps`][] for shared domain types
- [`prejectory.visualization`][] for optional visualization helpers

# Import guide

```python
import prejectory

from prejectory import datasets, runtime, processing, io, visualization
from prejectory.core import AgentCategory, DatasetSplit
```

"""

from importlib.metadata import PackageNotFoundError, version

__all__: list[str] = []

try:
    __version__ = version("prejectory")
except PackageNotFoundError:
    __version__ = "0+unknown"
