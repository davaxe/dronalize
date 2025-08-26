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

import json
from collections import deque
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import date

    from preprocessing.road_network.common import CategoryStr


def get_nuscenes_scenes_as_pandas(
    data_directory: Path,
) -> list[pd.DataFrame]:
    """Get all NuScenes scenes as pandas DataFrames.

    The directory should contain JSON files for various data types, e.g.,
    - attribute.json
    - calibrated_sensor.json
    - category.json
    - ego_pose.json
    - instance.json

    Args:
        data_directory: Path to the NuScenes data directory containing JSON files.

    Returns:
        A list of pandas DataFrames, each representing a scene with trajectory data.

    """
    nuscenes_data = NuscenesData(data_directory)
    return nuscenes_data.get_scenes_as_pandas()


class AttributeDict(TypedDict):
    """Nuscenes attribute data."""

    token: str
    name: str
    description: str


class CategoryDict(TypedDict):
    """Nuscenes category data."""

    token: str
    name: str
    description: str


class SceneDict(TypedDict):
    """Nuscenes scene data."""

    token: str
    log_token: str
    nbr_samples: int
    first_sample_token: str
    last_sample_token: str
    name: str
    description: str


class LogDict(TypedDict):
    """Nuscenes log data."""

    token: str
    logfile: str
    vehicle: str
    date_captured: date
    location: str


class SampleDict(TypedDict):
    """Nuscenes sample data."""

    token: str
    timestamp: int
    prev: str
    next: str
    scene_token: str


class SampleDataDict(TypedDict):
    """Nuscenes sample data."""

    token: str
    sample_token: str
    ego_pose_token: str
    timestamp: int
    prev: str
    next: str


class SampleAnnotationDict(TypedDict):
    """Nuscenes sample annotation data."""

    toke: str
    sample_token: str
    instance_token: str
    attribute_tokens: list[str]
    translation: list[float]
    size: list[float]
    rotation: list[float]
    prev: str
    next: str


class InstanceDict(TypedDict):
    """Nuscenes instance data."""

    token: str
    category_token: str
    nbr_annotations: int
    first_annotation_token: str
    last_annotation_token: str


class EgoPoseDict(TypedDict):
    """Nuscenes ego pose data."""

    token: str
    timestamp: int
    translation: list[float]
    rotation: list[float]


class NuscenesData:
    """Class to handle loading and accessing NuScenes data from JSON files."""

    def __init__(self, data_directory: Path | str) -> None:
        """Initialize with the path to the NuScenes data directory.

        The directory should contain JSON files for various data types, e.g.,
        - attribute.json
        - calibrated_sensor.json
        - category.json
        and so on.

        Args:
            data_directory: Path to the NuScenes data directory containing JSON
                files.

        """
        if isinstance(data_directory, Path):
            self.data_directory = data_directory
        elif isinstance(data_directory, str):
            self.data_directory = Path(data_directory)
        else:
            msg = "data directory must be a Path or a string representing the path."
            raise TypeError(msg)

    def get_scene_data_by_name(self, scene_name: str) -> pd.DataFrame:
        """Get the SceneData object for a specific scene name."""
        # Find the scene token by name
        for scene in self.scenes.values():
            if scene["name"] == scene_name:
                return _SceneData(scene["token"], self).to_pandas()
        msg = f"Scene with name '{scene_name}' not found."
        raise ValueError(msg)

    def get_scenes_as_pandas(self) -> list[pd.DataFrame]:
        """Get a list of all scenes as pandas DataFrames."""
        return list(self.get_scenes_as_pandas_iter())

    def get_scenes_as_pandas_iter(self) -> Iterable[pd.DataFrame]:
        """Get an iterable of all scenes as pandas DataFrames."""
        return (scene.to_pandas() for scene in self._get_scenes_iter())

    @cached_property
    def attributes(self) -> dict[str, AttributeDict]:
        """Mapping of attribute tokens to attribute objects (dicts)."""
        return self._to_token_dict(self._load_json("attribute"))

    @cached_property
    def categories(self) -> dict[str, CategoryDict]:
        """Mapping of category tokens to category objects (dicts)."""
        return self._to_token_dict(self._load_json("category"))

    @cached_property
    def ego_poses(self) -> dict[str, EgoPoseDict]:
        """Mapping of ego pose tokens to EgoPose objects (dicts)."""
        return self._to_token_dict(self._load_json("ego_pose"))

    @cached_property
    def scenes(self) -> dict[str, SceneDict]:
        """Mapping of scene tokens to Scene objects (dicts)."""
        return self._to_token_dict(self._load_json("scene"))

    @cached_property
    def instances(self) -> dict[str, InstanceDict]:
        """Mapping of instance tokens to Instance objects (dicts)."""
        return self._to_token_dict(self._load_json("instance"))

    @cached_property
    def logs(self) -> dict[str, LogDict]:
        """Mapping of log tokens to Log objects (dicts)."""
        return self._to_token_dict(self._load_json("log"))

    @cached_property
    def sample_annotations(self) -> dict[str, list[SampleAnnotationDict]]:
        """Mapping of sample tokens to lists of SampleAnnotation objects (dicts).

        There can be multiple annotations per sample, hence the list.
        """
        return self._to_token_dict_list(
            self._load_json("sample_annotation"),
            key="sample_token",
        )

    @cached_property
    def sample_data(self) -> dict[str, list[SampleDataDict]]:
        """Mapping of sample tokens to lists of SampleData objects (dicts).

        There can be multiple sample data entries per sample, hence the list.
        """
        return self._to_token_dict_list(
            self._load_json("sample_data"),
            key="sample_token",
        )

    @cached_property
    def samples(self) -> dict[str, SampleDict]:
        """Mapping of sample tokens to Sample objects (dicts)."""
        return self._to_token_dict(self._load_json("sample"))

    # --- Private methods ---

    def _get_scenes_iter(self) -> Iterable[_SceneData]:
        """Get an iterable of all scenes."""
        return (_SceneData(scene["token"], self) for scene in self.scenes.values())

    def _load_json(self, filename: str) -> list[dict[str, Any]]:
        file_path = self.data_directory / f"{filename}.json"
        with file_path.open("r") as file:
            return json.load(file)

    def _to_token_dict(
        self,
        data: list[dict[str, Any]],
        key: str = "token",
    ) -> dict[str, Any]:
        """Convert a list of dicts to a dict keyed by a specific key in the dict."""
        return {item[key]: cast("Any", item) for item in data}

    def _to_token_dict_list(
        self,
        data: list[dict[str, Any]],
        key: str = "token",
    ) -> dict[str, list[Any]]:
        """Convert a list of dicts to a dict keyed by a specific key in the dict.

        In contrast to `_to_token_dict`, this method groups items by the key,
        allowing for multiple items to share the same key. The values are lists
        of items that share the same key value.
        """
        result = {}
        for item in data:
            token = item[key]
            result.setdefault(token, []).append(cast("Any", item))
        return result


StatusStr = Literal[
    "moving",
    "stopped",
    "parked",
    "unknown",
]


class _Frame(TypedDict):
    """Each frame of trajectory data."""

    frame: int
    track_id: int
    x: float
    y: float
    agent_type: CategoryStr
    full_category: str
    status: StatusStr
    full_status: str
    map: str
    scene_name: str


class _SceneData:
    def __init__(self, scene_token: str, data: NuscenesData) -> None:
        self.scene_token: str = scene_token
        self.data: NuscenesData = data
        self.scene: SceneDict = data.scenes[scene_token]

        # Keep track of objects and their indices in the trajectory data.
        self._objects: dict[str, int] = {}
        # First index is reserved for the ego vehicle.
        self._ego_vehicle_index: int = 0

        self._data_rows: list[_Frame] = []
        self._process_scene()

    @property
    def location(self) -> str:
        """Get the location of the scene."""
        return self.data.logs[self.scene["log_token"]]["location"]

    def _process_scene(self) -> None:
        """Process the scene to extract trajectories."""
        samples: deque[str] = deque()
        samples.append(self.scene["first_sample_token"])
        sample_count: int = 0
        while samples:
            sample_token: str = samples.popleft()

            if sample_token == "":
                break

            self._process_sample(sample_token, sample_count)
            sample: SampleDict = self.data.samples[sample_token]
            samples.append(sample["next"])
            sample_count += 1

    def _process_sample(self, sample_token: str, sample_count: int = 0) -> None:
        sample_data: list[SampleDataDict] = self.data.sample_data[sample_token]
        ego_pose_token: str = sample_data[0]["ego_pose_token"]
        ego_position = self.data.ego_poses[ego_pose_token]["translation"]

        self._data_rows.append(
            _Frame(
                frame=sample_count,
                track_id=self._ego_vehicle_index,
                x=ego_position[0],
                y=ego_position[1],
                agent_type="car",
                full_category="vehicle.ego.car",
                status="moving",
                full_status="moving",
                map=self.location,
                scene_name=self.scene["name"],
            ),
        )

        self._data_rows.extend(
            self._get_sample_data_rows(sample_token, sample_count),
        )

    def _get_sample_data_rows(
        self,
        sample_token: str,
        frame: int,
    ) -> Iterable[_Frame]:
        if sample_token not in self.data.sample_annotations:
            return

        for annotation in self.data.sample_annotations[sample_token]:
            instance = annotation["instance_token"]
            yield _Frame(
                frame=frame,
                track_id=self._get_track_id(instance),
                x=annotation["translation"][0],
                y=annotation["translation"][1],
                agent_type=self._get_object_category(instance),
                full_category=self._get_full_object_category(instance),
                status=self._get_object_status(annotation["attribute_tokens"]),
                full_status=self._get_full_object_status(
                    annotation["attribute_tokens"],
                ),
                map=self.location,
                scene_name=self.scene["name"],
            )

    def _get_object_category(self, instance_token: str) -> CategoryStr:
        """Get the category of an object based on its instance token."""
        instance: InstanceDict = self.data.instances[instance_token]
        name: str = self.data.categories[instance["category_token"]]["name"]
        split_name = name.split(".")
        category = split_name[0]
        sub_type_dict = _CATEGORY_MAP.get(category, {})
        if isinstance(sub_type_dict, str):
            return sub_type_dict

        sub_type = split_name[1] if len(split_name) > 1 else "undefined"
        return sub_type_dict.get(sub_type, "undefined")

    def _get_full_object_category(self, instance_token: str) -> str:
        """Get the full category name of an object based on its instance token."""
        instance: InstanceDict = self.data.instances[instance_token]
        return self.data.categories[instance["category_token"]]["name"]

    def _get_track_id(self, instance_token: str) -> int:
        """Get the track ID for a given instance token."""
        if instance_token not in self._objects:
            index = len(self._objects) + 1
            self._objects[instance_token] = index
        return self._objects[instance_token]

    def _get_object_status(self, attribute_tokens: list[str]) -> StatusStr:
        """Get the status of an object based on its attribute tokens."""
        if not attribute_tokens:
            return "unknown"
        # For simplicity, we assume the first attribute is the most relevant.
        attribute = self.data.attributes[attribute_tokens[0]]["name"]
        attributes = attribute.split(".")
        if len(attributes) == 1:
            return _SINGLE_STATUS_MAP.get(attributes[0], "unknown")
        first, second, *_ = attributes
        return _STATUS_MAP.get(first, {}).get(second, "unknown")

    def _get_full_object_status(self, attribute_tokens: list[str]) -> str:
        """Get the full status of an object based on its attribute tokens."""
        if not attribute_tokens:
            return "unknown"

        # For simplicity, use the first attribute token (in all observed cases
        # only one exists).
        return self.data.attributes[attribute_tokens[0]]["name"]

    def to_pandas(self) -> pd.DataFrame:
        """Convert the trajectory data to a pandas DataFrame."""
        return pd.DataFrame(self._data_rows)


_SINGLE_STATUS_MAP: dict[str, StatusStr] = {"dummy": "unknown"}

_STATUS_MAP: dict[str, dict[str, StatusStr]] = {
    "vehicle": {
        "moving": "moving",
        "stopped": "stopped",
        "parked": "parked",
    },
    "pedestrian": {
        "moving": "moving",
        "standing": "stopped",
        "sitting_lying_down": "stopped",
    },
    "cycle": {
        "with_rider": "moving",
        "without_rider": "stopped",
    },
}

_CATEGORY_MAP: dict[str, dict[str, CategoryStr] | CategoryStr] = {
    "vehicle": {
        "car": "car",
        "van": "van",
        "construction": "van",  # mapped to "van" as in original logic
        "motorcycle": "motorcycle",
        "bicycle": "bicycle",
        "bus": "bus",
        "truck": "truck",
        "trailer": "trailer",
        "ego": "car",
    },
    "human": "pedestrian",
    "static": "static_object",
    "static_object": "static_object",
    "movable_object": "movable_object",
    "animal": "animal",
}

if __name__ == "__main__":
    data_dir = Path("data/v1.0-trainval")
    nuscenes_data = NuscenesData(data_dir)
    # Sort based on track_id and then frame
    pandas_scenes = nuscenes_data.get_scenes_as_pandas()
    print(len(pandas_scenes))
    scene_trajectories = pandas_scenes[0]
    print(scene_trajectories.head())
