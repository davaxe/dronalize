from pathlib import Path

from typing_extensions import override

from dronalize.core import LoaderConfig
from dronalize.datasets.nuscenes.loader import NuScenesLoader


class VodLoader(NuScenesLoader):
    """View-of-Delft dataset processor.

    This shares the same base processing logic as NuScenesProcessor but with
    specific category filtering. For further details se the nuScenes processor.

    """

    def __init__(
        self,
        data_dir: Path | str,
        loader_config: LoaderConfig | None = None,
    ) -> None:
        """Initialize the processor.

        Parameters
        ----------
        data_dir : Path or str
            Directory of the trajectory data JSON files.
        loader_config : LoaderConfig, optional
            Custom configuration, or default if None.

        """
        super().__init__(data_dir, loader_config=loader_config)
        self._full_category_contains = [
            "vehicle.ego",
        ]

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(5, 30, 0.1).with_window(5).with_filtering(require_frames=[4])
