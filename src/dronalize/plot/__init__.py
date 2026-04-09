"""Optional plotting helpers for trajectory and map inspection.

## Import guide

```python
from dronalize.plot import plot_map_graph, plot_trajectories, plot_trajectories_on_map
```

Plotting helpers are loaded lazily so importing `dronalize.plot` does not pull
in optional visualization dependencies until one of the plotting functions is
used.

The package exposes three focused entry points:

- [`plot_trajectories`][dronalize.plot.plot_trajectories] for trajectory-only
  inspection
- [`plot_map_graph`][dronalize.plot.plot_map_graph] for lane-graph or road-graph
  inspection
- [`plot_trajectories_on_map`][dronalize.plot.plot_trajectories_on_map] for
  combined trajectory and map overlays

## Related modules

- [`dronalize.core.scene`][] for scene types commonly visualized here
- [`dronalize.core.maps`][] for map graph types consumed by map plots
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.plot.map import plot_map_graph
    from dronalize.plot.overlay import plot_trajectories_on_map
    from dronalize.plot.trajectory import plot_trajectories

__all__ = ["plot_map_graph", "plot_trajectories", "plot_trajectories_on_map"]

_EXPORTS: dict[str, tuple[str, str]] = {
    "plot_map_graph": ("dronalize.plot.map", "plot_map_graph"),
    "plot_trajectories": ("dronalize.plot.trajectory", "plot_trajectories"),
    "plot_trajectories_on_map": ("dronalize.plot.overlay", "plot_trajectories_on_map"),
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
