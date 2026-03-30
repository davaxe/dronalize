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


class ExiDLoader(LevelXDataLoader):
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
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitRequest | None = None,
    ) -> None:
        """Initialize the exiD loader."""
        data_root = Path(data_root)
        super().__init__(
            data_root / "data",
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )


if __name__ == "__main__":
    from dronalize.datasets.exid import DESCRIPTOR
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env

    root = resolve_dataset_root_from_env("exid")
    _ = debug_descriptor(DESCRIPTOR, root)
