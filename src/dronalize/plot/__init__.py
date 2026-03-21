"""Plot package — optional visualisation utilities for trajectories and map graphs.

Requires the `plot` extra::

    pip install dronalize[plot]
"""

from dronalize._internal._compat import require_optional

# Since this whole package is optional and require the altair dependency (from
# "plot" extra), it is validated at the package level so all modules can assume
# it is present.
_ = require_optional("altair", extra="plot")

from dronalize.plot.map import plot_map_graph  # noqa: E402
from dronalize.plot.overlay import plot_trajectories_on_map  # noqa: E402
from dronalize.plot.trajectory import plot_trajectories  # noqa: E402

__all__ = [
    "plot_map_graph",
    "plot_trajectories",
    "plot_trajectories_on_map",
]
