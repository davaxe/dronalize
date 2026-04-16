"""Framework-neutral readers for persisted dataset outputs.

All readers in this package share the same output contract: they yield
[`SceneRecord`][dronalize.io.SceneRecord] instances, which are
storage-agnostic in-memory representations of scene records.

## Import guide

```python
from dronalize.io.readers import DatasetReader, MDSReader, MDSReaderInitArgs, PickleReader
```

## Related modules

- [`dronalize.io`][] for storage contracts and export configuration
- [`dronalize.io.adapters`][] for optional adapter layers built on top of
  readers
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.io.base import DatasetReader
    from dronalize.io.readers.mds import MDSReader, MDSReaderInitArgs
    from dronalize.io.readers.pickle import PickleReader

__all__ = ["DatasetReader", "MDSReader", "MDSReaderInitArgs", "PickleReader"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DatasetReader": ("dronalize.io.base", "DatasetReader"),
    "MDSReader": ("dronalize.io.readers.mds", "MDSReader"),
    "MDSReaderInitArgs": ("dronalize.io.readers.mds", "MDSReaderInitArgs"),
    "PickleReader": ("dronalize.io.readers.pickle", "PickleReader"),
}


def __getattr__(name: str) -> object:
    """Resolve optional reader exports lazily."""
    if name not in _EXPORTS:
        msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    module_name, export_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), export_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy reader exports during interactive discovery."""
    return sorted(set(globals()) | set(__all__))
