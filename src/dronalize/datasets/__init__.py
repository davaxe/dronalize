"""Dataset registry and dataset-descriptor surface.

## Import guide

```python
from dronalize.datasets import available, get, register, DatasetSpec
```

This package is the canonical entry point for dataset discovery and custom
dataset registration.

- [`available`][dronalize.datasets.available] lists registered and built-in
  dataset keys
- [`get`][dronalize.datasets.get] resolves one dataset descriptor by name
- [`register`][dronalize.datasets.register] adds a custom
  [`DatasetSpec`][dronalize.datasets.DatasetSpec] to the active registry
- [`DatasetSpec`][dronalize.datasets.DatasetSpec] and
  [`DatasetCapabilities`][dronalize.datasets.DatasetCapabilities] describe the
  processing contract that the runtime planner consumes

Built-in dataset package roots remain importable, but the registry is the main
public interface for selecting datasets.

## Related modules

- [`dronalize.processing.loading`][] for the advanced loader API
- [`dronalize.runtime`][] for planning APIs that consume
  [`DatasetSpec`][dronalize.datasets.DatasetSpec]
"""

from dronalize.datasets.registry import (
    DatasetCapabilities,
    DatasetSpec,
    RuntimeContext,
    available,
    get,
    register,
)

__all__ = [
    "DatasetCapabilities",
    "DatasetSpec",
    "RuntimeContext",
    "available",
    "get",
    "register",
]
