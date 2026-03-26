from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.datasets.i80.loader import I80Loader

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.categories import DatasetSplit
    from dronalize.config.loader import LoaderConfig
    from dronalize.config.map import MapConfig
    from dronalize.config.split import SplitRequest


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
        split_request: SplitRequest | None = None,
    ) -> None:
        """Initialize the US-101 loader."""
        super().__init__(
            data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )
