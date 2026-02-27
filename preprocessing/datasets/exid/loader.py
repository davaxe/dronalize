from __future__ import annotations

from preprocessing.common.trajectory_utils.levelx import XLevelDataLoader


class ExiDLoader(XLevelDataLoader):
    """Trajectory data loader for the exiD dataset.

    The exiD (extracted from Drone) dataset was recorded at highway exits and entries in Germany.
    It contains naturalistic vehicle trajectories extracted from drone footage, covering a variety
    of traffic participants including cars, trucks, buses, and motorcycles interacting at
    on-ramps and off-ramps.

    This loader inherits all trajectory processing logic from `XLevelDataLoader`, as the
    exiD dataset follows the same CSV format used across the X-level dataset family.
    """
