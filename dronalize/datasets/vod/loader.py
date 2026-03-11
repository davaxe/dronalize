from pathlib import Path

from typing_extensions import override

from dronalize.config import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.nuscenes.loader import NuScenesLoader


class VodLoader(NuScenesLoader):
    """Loader for the View-of-Delft dataset.

    This shares the same base loading logic as `NuScenesLoader`, but with
    dataset-specific category filtering. For further details, see
    `NuScenesLoader`.

    """

    def __init__(
        self,
        data_dir: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
    ) -> None:
        """Initialize the dataset loader.

        Parameters
        ----------
        data_dir : Path or str
            Directory of the trajectory data JSON files.
        loader_config : , optional
            Custom configuration, or default if None.

        """
        super().__init__(data_dir, loader_config=loader_config, map_config=map_config)
        self._full_category_contains = [
            "vehicle.ego",
        ]

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=5, output_len=30, sample_time=0.1)
            .with_window(5)
            .with_filtering(require_frames=[4])
        )
