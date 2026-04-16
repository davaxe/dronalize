"""Optional framework adapters built on top of persisted dataset readers.

## Import guide

```python
from dronalize.io.adapters import HeteroSceneDataset, TorchSceneDataset
```

This package groups higher-level dataset adapters for downstream ML code. The
exports are loaded lazily so optional dependencies such as Torch or
Torch-Geometric are only imported when their adapters are actually requested.

Use [`dronalize.io.readers`][] when you want framework-neutral records. Use this
package when you want those records exposed through Torch or PyG dataset
surfaces.

## Related modules

- [`dronalize.io.readers`][] for framework-neutral persisted dataset readers
- [`dronalize.io`][] for storage contracts and export configuration
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.io.adapters.pyg import (
        HeteroSceneDataset,
        IterableHeteroSceneDataset,
        collate_hetero_with_time_padding,
    )
    from dronalize.io.adapters.torch import (
        IterableTorchSceneDataset,
        TorchSceneDataset,
        TorchSceneRecord,
    )

__all__ = [
    "HeteroSceneDataset",
    "IterableHeteroSceneDataset",
    "IterableTorchSceneDataset",
    "TorchSceneDataset",
    "TorchSceneRecord",
    "collate_hetero_with_time_padding",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "IterableTorchSceneDataset": ("dronalize.io.adapters.torch", "IterableTorchSceneDataset"),
    "TorchSceneDataset": ("dronalize.io.adapters.torch", "TorchSceneDataset"),
    "TorchSceneRecord": ("dronalize.io.adapters.torch", "TorchSceneRecord"),
    "HeteroSceneDataset": ("dronalize.io.adapters.pyg", "HeteroSceneDataset"),
    "IterableHeteroSceneDataset": ("dronalize.io.adapters.pyg", "IterableHeteroSceneDataset"),
    "collate_hetero_with_time_padding": (
        "dronalize.io.adapters.pyg",
        "collate_hetero_with_time_padding",
    ),
}


def __getattr__(name: str) -> object:
    """Resolve optional adapter exports lazily."""
    if name not in _EXPORTS:
        msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    module_name, export_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), export_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy adapter exports during interactive discovery."""
    return sorted(set(globals()) | set(__all__))
