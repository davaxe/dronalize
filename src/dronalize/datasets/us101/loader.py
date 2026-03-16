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
        *,
        lane_change_ratio: float | None = 1,
    ) -> None:
        """Initialize the US101 dataset loader.

        It is possible to rebalance the dataset by adjusting the number of lane
        changing agents compared to non-lane changing agents. This can be done
        by setting the `lane_change_ratio` parameter. For example, a ratio of
        0.5 would result in half as many lane changing agents as non-lane
        changing agents. Typically highway datasets are heavily imbalanced
        towards non-lane changing agents, which means that a high ratio con
        result in way less total data.

        Parameters
        ----------
        data_root : Path
            Path to root of the US101 dataset, containing subdirectories of data files.
        loader_config : , optional
            Loader configuration. If None, the default configuration is used.
        lane_change_ratio : float, optional
            Ratio for rebalancing highway agents. If None, no rebalancing will
            be applied. Default is 1.0, i.e. same number of lane changes as
            non-lane changes.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            data_root,
            loader_config=loader_config,
            map_config=map_config,
            lane_change_ratio=lane_change_ratio,
            splits=splits,
        )
