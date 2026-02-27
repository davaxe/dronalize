from __future__ import annotations

from preprocessing.common.loaders.xlevel import XLevelDataLoader


class InDLoader(XLevelDataLoader):
    """Trajectory data loader for the inD dataset.

    The inD (intersections Drone) dataset was recorded at urban intersections in Germany using
    drone footage. It contains naturalistic trajectories of various traffic participants —
    including cars, trucks, bicycles, and pedestrians — navigating through signalised and
    unsignalised intersections.

    This loader inherits all trajectory processing logic from `XLevelDataLoader`, as the
    inD dataset follows the same CSV format used across the X-level dataset family.
    """
