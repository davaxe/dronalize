from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.datasets.common.xlevel_loader import XLevelDataLoader

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.config import LoaderConfig
    from dronalize.config.map import MapConfig


class ExiDLoader(XLevelDataLoader):
    """Trajectory data loader for the exiD dataset.

    The exiD (extracted from Drone) dataset was recorded at highway exits and
    entries in Germany. It contains naturalistic vehicle trajectories extracted
    from drone footage, covering a variety of traffic participants including
    cars, trucks, buses, and motorcycles interacting at on-ramps and off-ramps.

    This loader inherits all trajectory processing logic from
    `XLevelDataLoader`, as the exiD dataset follows the same CSV format used
    across the X-level dataset family.

    """

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
    ) -> None:
        """Initialize the trajectory data loader for the exiD dataset.

        Parameters
        ----------
        data_root : Path
            Path to root of the exiD dataset.
        loader_config : , optional
            Loader configuration. If None, the default configuration is used.

        """
        data_root = self._normalize_data_root(data_root)
        super().__init__(data_root / "data", loader_config=loader_config, map_config=map_config)
