from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.datasets.i80.loader import I80Loader

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from dronalize.categories import DatasetSplit
    from dronalize.config import LoaderConfig
    from dronalize.config.map import MapConfig


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
    ) -> None:
        """Initialize the US101 dataset loader.

        Parameters
        ----------
        data_root : Path
            Path to root of the US101 dataset, containing subdirectories of data files.
        loader_config : , optional
            Loader configuration. If None, the default configuration is used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            data_root,
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
        )
