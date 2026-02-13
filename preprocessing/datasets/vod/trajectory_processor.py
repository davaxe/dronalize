from pathlib import Path
from typing import override

from preprocessing.core.interface import ProcessorConfig
from preprocessing.datasets.nuscenes.trajectory_processor import NuScenesProcessor


class VodProcessor(NuScenesProcessor):
    """View-of-Delft dataset processor.

    This shares the same base processing logic as NuScenesProcessor but with
    specific category filtering. For further details se the nuScenes processor.
    """

    def __init__(
        self,
        data_directory: Path | str,
        processor_config: ProcessorConfig | None = None,
    ) -> None:
        """Initialize the processor.

        Args:
            data_directory: directory of tha trajectory data JSON files.
            processor_config: custom configuration or default if None.

        """
        super().__init__(data_directory, processor_config=processor_config)
        self._full_category_contains = [
            "vehicle.ego",
        ]

    @override
    def _default_config(self) -> ProcessorConfig:
        return ProcessorConfig(5, 30, 0.1).window_parameters(5)


if __name__ == "__main__":
    import time

    # Update this path to your actual data location
    data_dir = Path(
        "/home/west/Developer/behavior-prediction/datasets/vod/v1.0-trainval/"
    )

    # Check if directory exists to avoid FileNotFound errors in example
    if data_dir.exists():
        start_time = time.perf_counter()
        processor = VodProcessor(data_directory=data_dir)
        for _scene in processor.scenes_iter():
            if _scene.scene_number % 200 == 0:
                print(f"Processing scene number: {_scene.scene_number}")
    else:
        print(f"Path not found: {data_dir}")
