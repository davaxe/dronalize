"""Plot package: optional visualization utilities for trajectories and map graphs.

Requires the `plot` extra::

    pip install dronalize[plot]
"""

from dronalize._internal._optional import require_optional

# Validate the optional plotting dependency once at package import time.
_ = require_optional("altair", extra="plot")

from dronalize.plot.map import plot_map_graph  # noqa: E402
from dronalize.plot.overlay import plot_trajectories_on_map  # noqa: E402
from dronalize.plot.trajectory import plot_trajectories  # noqa: E402

__all__ = [
    "plot_map_graph",
    "plot_trajectories",
    "plot_trajectories_on_map",
]
