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

from typing import TYPE_CHECKING

from dronalize._lazy import lazy_dir, resolve_lazy_export

if TYPE_CHECKING:
    from dronalize.plot.scene import AspectMode, plot_scene
    from dronalize.plot.theme import PlotTheme

__all__ = ["AspectMode", "PlotTheme", "plot_scene"]

__lazy_exports__: dict[str, tuple[str, str]] = {
    "plot_scene": ("dronalize.plot.scene", "plot_scene"),
    "AspectMode": ("dronalize.plot.scene", "AspectMode"),
    "PlotTheme": ("dronalize.plot.theme", "PlotTheme"),
}


def __getattr__(name: str) -> object:
    """Resolve plotting helpers lazily to avoid importing optional deps eagerly."""
    return resolve_lazy_export(globals(), __lazy_exports__, module_name=__name__, name=name)


def __dir__() -> list[str]:
    """Expose lazy plotting exports during interactive discovery."""
    return lazy_dir(globals(), exported_names=list(__lazy_exports__))
