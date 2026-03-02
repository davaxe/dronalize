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
        data_directory: Path | str,
        loader_config: LoaderConfig | None = None,
    ) -> None:
        """Initialize the processor.

        Parameters
        ----------
        data_directory : Path or str
            Directory of the trajectory data JSON files.
        loader_config : LoaderConfig, optional
            Custom configuration, or default if None.

        """
        super().__init__(data_directory, loader_config=loader_config)
        self._full_category_contains = [
            "vehicle.ego",
        ]

    @override
    def default_config(self) -> LoaderConfig:
        return LoaderConfig(5, 30, 0.1).window_parameters(5)


if __name__ == "__main__":
    import time

    # Update this path to your actual data location
    data_dir = Path("/home/west/Developer/behavior-prediction/datasets/vod/v1.0-trainval/")

    # Check if directory exists to avoid FileNotFound errors in example
    if data_dir.exists():
        start_time = time.perf_counter()
        processor = VodLoader(data_directory=data_dir)
        for _scene in processor.scenes():
            if _scene.scene_number % 200 == 0:
                print(f"Processing scene number: {_scene.scene_number}")
    else:
        print(f"Path not found: {data_dir}")
