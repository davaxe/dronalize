from __future__ import annotations

from preprocessing.common.trajectory_utils.levelx import XLevelDataLoader


class RounDLoader(XLevelDataLoader):
    """Trajectory data loader for the rounD dataset.

    The rounD (roundabouts Drone) dataset was recorded at roundabouts in Germany using drone
    footage. It contains naturalistic trajectories of various traffic participants, including
    cars, trucks, buses, motorcycles, bicycles, and pedestrians navigating through roundabouts
    of varying sizes and layouts.

    This loader inherits all trajectory processing logic from `XLevelDataLoader`, as the
    rounD dataset follows the same CSV format used across the X-level dataset family.
    """
