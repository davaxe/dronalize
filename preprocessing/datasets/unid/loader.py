from __future__ import annotations

from preprocessing.common.loaders.xlevel import XLevelDataLoader


class UniDLoader(XLevelDataLoader):
    """Trajectory data loader for the uniD dataset.

    The uniD (urban intersections Drone) dataset was recorded at urban intersections in Germany
    using drone footage. It contains naturalistic trajectories of various traffic participants,
    including cars, trucks, buses, motorcycles, bicycles, and pedestrians navigating through
    signalised and unsignalised intersections.

    This loader inherits all trajectory processing logic from `XLevelDataLoader`, as the
    uniD dataset follows the same CSV format used across the X-level dataset family.
    """
