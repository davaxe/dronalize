from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.datasets.common.levelx_loader import LevelXDataLoader

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.categories import DatasetSplit
    from dronalize.config import LoaderConfig
    from dronalize.config.map import MapConfig


class InDLoader(LevelXDataLoader):
    """Trajectory data loader for the inD dataset.

    The inD (intersections Drone) dataset was recorded at urban intersections in
    Germany using drone footage. It contains naturalistic trajectories of
    various traffic participants — including cars, trucks, bicycles, and
    pedestrians — navigating through signalised and unsignalised intersections.

    This loader inherits all trajectory processing logic from
    `XLevelDataLoader`, as the inD dataset follows the same CSV format used
    across the X-level dataset family.

    """

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        """Initialize the trajectory data loader for the inD dataset.

        Parameters
        ----------
        data_root : Path
            Path to root of the inD dataset.
        loader_config : , optional
            Loader configuration. If None, the default configuration is used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            Path(data_root) / "data",
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
        )
