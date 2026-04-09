"""Framework-neutral readers for persisted dataset outputs.

## Import guide

```python
from dronalize.io.readers import MDSReader, MDSReaderInitArgs
```

Reader symbols are loaded lazily so importing this package does not eagerly pull
in optional storage dependencies. Use this package when you want NumPy-backed
records rather than framework-specific dataset adapters.

## Related modules

- [`dronalize.io`][] for storage contracts and export configuration
- [`dronalize.io.adapters`][] for optional adapter layers built on top of
  readers
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.io.readers.mds import MDSReader, MDSReaderInitArgs

__all__ = ["MDSReader", "MDSReaderInitArgs"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "MDSReader": ("dronalize.io.readers.mds", "MDSReader"),
    "MDSReaderInitArgs": ("dronalize.io.readers.mds", "MDSReaderInitArgs"),
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
