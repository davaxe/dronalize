from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.datasets.shared.levelx_loader import LevelXDataLoader

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.categories import DatasetSplit
    from dronalize.processing.ingest.config import LoaderConfig
    from dronalize.processing.ingest.splits import SplitRequest
    from dronalize.processing.maps.config import MapConfig


class RounDLoader(LevelXDataLoader):
    """Trajectory data loader for the rounD dataset.

    The rounD (roundabouts Drone) dataset was recorded at roundabouts in Germany
    using drone footage. It contains naturalistic trajectories of various
    traffic participants, including cars, trucks, buses, motorcycles, bicycles,
    and pedestrians navigating through roundabouts of varying sizes and layouts.

    This loader inherits all trajectory processing logic from
    `XLevelDataLoader`, as the rounD dataset follows the same CSV format used
    across the X-level dataset family.

    """

    def __init__(
        self,
        data_root: str | Path,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitRequest | None = None,
    ) -> None:
        """Initialize the rounD loader."""
        super().__init__(
            Path(data_root) / "data",
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )
