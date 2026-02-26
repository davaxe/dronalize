from __future__ import annotations

from typing import TYPE_CHECKING

from preprocessing.datasets.i80.loader import I80Loader

if TYPE_CHECKING:
    from pathlib import Path

    from preprocessing.core.interface import LoaderConfig


class US101Loader(I80Loader):
    """Scene loader for the US-101 dataset.

    For more details see `I80Loader`, as the US-101 dataset has the same format and structure.
    """

    def __init__(
        self,
        data_dir: Path,
        config: LoaderConfig | None = None,
        lane_change_ratio: float | None = 1,
    ) -> None:
        """Initialize the US101 dataset loader.

        It is possible to rebalance the dataset by adjusting the number of lane changing agents
        compared to non-lane changing agents. This can be done by setting the `lane_change_ratio`
        parameter. For example, a ratio of 0.5 would result in half as many lane changing agents as
        non-lane changing agents. Typically highway datasets are heavily imbalanced towards non-lane
        changing agents, which means that a high ratio con result in way less total data.

        Args:
            data_dir: Path to root of the I80 dataset, containing subdirectores of data files.
            config: Optional processor configuration. If None, default configuration will be used.
            lane_change_ratio: Optional ratio for rebalancing highway agents. If None, no
                rebalancing will be applied. Default is 1.0, i.e. same number of lane changes as
                non-lane changes.

        """
        super().__init__(data_dir, config, lane_change_ratio)
