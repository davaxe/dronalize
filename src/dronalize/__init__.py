"""Top-level package namespace for the `dronalize` library.

This module is intentionally small. It does not provide convenience aliases
for the main Python API. Instead, the public surface is organized around
explicit package namespaces so imports stay predictable and module ownership
remains clear.

Use the package namespaces directly:

- [`dronalize.datasets`][] for dataset lookup, descriptors, and registration
- [`dronalize.runtime`][] for config resolution, planning, and run-state models
- [`dronalize.processing`][] for processing config and grouped processing modules
- [`dronalize.io`][] for export config, manifests, readers, and adapters
- [`dronalize.core.scene`][] and [`dronalize.core.maps`][] for shared domain types
- [`dronalize.visualization`][] for optional visualization helpers

# Import guide

```python
import dronalize

from dronalize import datasets, runtime, processing, io, visualization
from dronalize.core import AgentCategory, DatasetSplit
```

"""

from importlib.metadata import version

__all__: list[str] = []

__version__: str = version("dronalize")
