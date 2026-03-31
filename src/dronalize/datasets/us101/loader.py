from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.datasets.i80.loader import I80Loader

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.core.categories import DatasetSplit
    from dronalize.processing.ingest.config import LoaderConfig
    from dronalize.processing.ingest.splits import SplitConfig
    from dronalize.processing.maps.config import MapConfig


class US101Loader(I80Loader):
    """Scene loader for the US-101 dataset.

    For more details see `I80Loader`, as the US-101 dataset has the same format
    and structure.

    """

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
    ) -> None:
        """Initialize the US-101 loader."""
        super().__init__(
            data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )


if __name__ == "__main__":
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env
    from dronalize.datasets.us101 import DESCRIPTOR

    root = resolve_dataset_root_from_env("us101")
    _ = debug_descriptor(DESCRIPTOR, root)
