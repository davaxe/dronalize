"""Optional plotting helpers for scene inspection.

## Import guide

```python
from dronalize.plot import plot_scene
```

Plotting helpers are loaded lazily so importing `dronalize.plot` does not pull
in optional visualization dependencies until plotting is requested.

The package exposes one scene-oriented entrypoint:

- [`plot_scene`][dronalize.plot.plot_scene] for plotting a
  [`Scene`][dronalize.core.scene.Scene] or
  [`SceneRecord`][dronalize.io.SceneRecord]

## Related modules

- [`dronalize.core.scene`][] for scene containers commonly visualized here
- [`dronalize.io`][] for persisted scene-record containers also accepted here
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from dronalize.plot.scene import plot_scene

AspectMode = Literal["auto", "equal"]
"""Aspect-ratio modes accepted by [`plot_scene`][dronalize.plot.plot_scene]."""

__all__ = ["AspectMode", "plot_scene"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "plot_scene": ("dronalize.plot.scene", "plot_scene"),
    "AspectMode": ("dronalize.plot.scene", "AspectMode"),
}


def __getattr__(name: str) -> object:
    """Resolve plotting helpers lazily to avoid importing optional deps eagerly."""
    if name not in _EXPORTS:
        msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg)

    module_name, export_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), export_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy plotting exports during interactive discovery."""
    return sorted(set(globals()) | set(__all__))
