"""Plotting utilities for map graphs and trajectories.

This subpackage is optional and requires `matplotlib` (and optionally
`altair` for trajectory plots).  Install with::

    pip install dronalize[plot]
"""

from preprocessing.common.plotting.map import plot_map_graph
from preprocessing.common.plotting.trajectory import plot_trajectories

__all__ = [
    "plot_map_graph",
    "plot_trajectories",
]
