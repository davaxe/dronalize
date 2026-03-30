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
        split_request: SplitRequest | None = None,
    ) -> None:
        """Initialize the inD loader."""
        super().__init__(
            Path(data_root) / "data",
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )


if __name__ == "__main__":
    from dronalize.datasets.ind import DESCRIPTOR
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env

    root = resolve_dataset_root_from_env("ind")
    _ = debug_descriptor(DESCRIPTOR, root)
