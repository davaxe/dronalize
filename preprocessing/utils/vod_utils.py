# Copyright 2024-2025, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pandas.core.api import DataFrame as DataFrame

from preprocessing.utils import nuscenes_utils

if TYPE_CHECKING:
    import pandas as pd


def get_vod_scenes_as_pandas(
    data_directory: Path,
) -> list[pd.DataFrame]:
    """Get all VOD scenes as pandas DataFrames.

    Args:
        data_directory: Path to the VOD data directory containing JSON files.

    Returns:
        A list of pandas DataFrames, each representing a VOD scene with trajectory
            data.

    """
    vod_data = VODData(data_directory)
    return vod_data.get_scenes_as_pandas()


class VODData(nuscenes_utils.NuscenesData):
    """Class to handle VOD data loading from JSON files."""

    def __init__(self, data_directory: Path | str) -> None:
        """Initialize with the path to the VOD data directory.

        The directory should contain JSON files for various data types, e.g.,
        - attribute.json
        - calibrated_sensor.json
        - category.json
        and so on.

        Args:
            data_directory: Path to the NuScenes data directory containing JSON
                files.

        """
        super().__init__(data_directory)

    def get_scenes_as_pandas_iter(self) -> nuscenes_utils.Iterable[DataFrame]:
        """Get an iterable of all scenes as pandas DataFrames."""
        for scene in self._get_scenes_iter():
            # Remove all rows where full_category is not vehicle.ego as it is
            # duplicated
            pandas_scene = scene.to_pandas()
            pandas_scene = pandas_scene[
                pandas_scene["full_category"] != "vehicle.ego"
            ]
            yield pandas_scene


if __name__ == "__main__":
    data_dir = Path("data/vod_official/v1.0-trainval")
    nuscenes_data = VODData(data_dir)
    # Sort based on track_id and then frame
    pandas_scenes = nuscenes_data.get_scenes_as_pandas()
    print(len(pandas_scenes))
    scene_trajectories = pandas_scenes[0]
    print(scene_trajectories.head())
