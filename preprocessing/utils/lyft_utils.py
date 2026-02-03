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

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

try:
    import zarr
except ImportError:
    HAS_ZARR = False
else:
    HAS_ZARR = True

if TYPE_CHECKING:
    from collections.abc import Iterable

    import zarr

    from preprocessing.road_network.common import CategoryStr


def get_lyft_scenes_as_pandas_lazy(
    directory_path: Path | str,
    start: int = 0,
    batch_size: int | None = 1000,
) -> Iterable[pd.DataFrame]:
    """Get Lyft scenes as a lazy iterable of pandas DataFrames."""
    if not HAS_ZARR:
        msg = (
            "Zarr is required to read the Lyft dataset. "
            "Please install it with `pip install zarr`."
        )
        raise ImportError(msg)

    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    scenes = zarr.open_array(
        directory_path / "scenes",
        mode="r",
    )
    frames = zarr.open_array(
        directory_path / "frames",
        mode="r",
    )
    agents = zarr.open_array(
        directory_path / "agents",
        mode="r",
    )

    return _scenes_to_pandas_batches(scenes, frames, agents, start, batch_size)


def get_lyft_scenes_as_pandas_list(
    directory_path: Path | str,
    start: int = 0,
    batch_size: int | None = 1000,
) -> list[pd.DataFrame]:
    """Get Lyft scenes as a list of pandas DataFrames."""
    if not HAS_ZARR:
        msg = (
            "Zarr is required to read the Lyft dataset. "
            "Please install it with `pip install zarr`."
        )
        raise ImportError(msg)

    if isinstance(directory_path, str):
        directory_path = Path(directory_path)

    scenes = zarr.open_array(
        directory_path / "scenes",
        mode="r",
    )
    frames = zarr.open_array(
        directory_path / "frames",
        mode="r",
    )
    agents = zarr.open_array(
        directory_path / "agents",
        mode="r",
    )

    return list(_scenes_to_pandas_batches(scenes, frames, agents, start, batch_size))


class _Frame(TypedDict):
    """Each frame of trajectory data."""

    frame: int
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    yaw: float
    category: CategoryStr
    scene_name: str


@dataclass
class LyftScene:
    """A single scene in the Lyft dataset."""

    scene_data: npt.NDArray

    @property
    def scene_name(self) -> str:
        """Return the name of the scene."""
        return str(self.scene_data["host"])

    @property
    def frame_interval(self) -> tuple[int, int]:
        """Return the frame indices of the scene."""
        start, end = self.scene_data["frame_index_interval"]
        return (int(start), int(end))


@dataclass(init=False)
class LyftFrames:
    """A collection of frames in a specific interval."""

    frames: npt.NDArray

    def __init__(
        self,
        frames: npt.NDArray,
        interval: tuple[int, int] | None = None,
        frame_offset: int = 0,
    ) -> None:
        """Initialize LyftFrames with frames and an optional interval."""
        if interval is None:
            self.frames = frames
        else:
            start, end = interval
            self.frames = frames[start - frame_offset : end - frame_offset]

    @property
    def agent_intervals(self) -> npt.NDArray:
        """Return the intervals of the agents in the frames."""
        return self.frames["agent_index_interval"]

    def agent_interval(self, frame_index: int) -> tuple[int, int]:
        """Return the agent interval for a specific frame index."""
        return tuple(self.frames["agent_index_interval"][frame_index])

    def ego_data(self, frame_index: int) -> _Frame:
        """Return the ego data for a specific frame index."""
        frame = self.frames[frame_index]
        return _Frame(
            frame=frame_index,
            track_id=-1,
            x=float(frame["ego_translation"][0]),
            y=float(frame["ego_translation"][1]),
            vx=0,
            vy=0,
            yaw=0,
            category="car",
            scene_name="",
        )


@dataclass(init=False)
class LyftAgents:
    """A collection of agents in a specific interval."""

    agents: npt.NDArray

    def __init__(
        self,
        agent_array: npt.NDArray,
        interval: tuple[int, int] | None = None,
        agent_offset: int = 0,
    ) -> None:
        """Initialize LyftAgents with an agent array and an interval."""
        if interval is None:
            self.agents = agent_array
        else:
            start, end = interval
            self.agents = agent_array[start - agent_offset : end - agent_offset]


def _agents_iterator(
    agents: npt.NDArray,
    frames: LyftFrames,
    agent_offset: int = 0,
) -> Iterable[LyftAgents]:
    """Iterate over agents in the frames."""
    for frame_index in range(len(frames.frames)):
        start, end = frames.agent_interval(frame_index)
        yield LyftAgents(
            agent_array=agents,
            interval=(start, end),
            agent_offset=agent_offset,
        )


def _scenes_to_pandas_batches(
    scenes: zarr.Array,
    frames: zarr.Array,
    agents: zarr.Array,
    start: int = 0,
    batch_size: int | None = 1000,
) -> Iterable[pd.DataFrame]:
    current = start
    total_scenes: int = scenes.shape[0]

    if batch_size is None:
        batch_size = total_scenes

    while current < total_scenes:
        end = min(current + batch_size, total_scenes)
        scene_interval = (current, end)
        yield from _scenes_to_pandas(
            scenes=scenes,
            frames=frames,
            agents=agents,
            scene_interval=scene_interval,
        )
        current += batch_size


def _scenes_to_pandas(
    scenes: zarr.Array,
    frames: zarr.Array,
    agents: zarr.Array,
    scene_interval: tuple[int, int] | None = None,
) -> Iterable[pd.DataFrame]:
    if scene_interval is not None:
        start, end = scene_interval
        scenes_np = cast("npt.NDArray", scenes[start:end])
    else:
        scenes_np = cast("npt.NDArray", scenes[:])

    frame_start = np.min(
        scenes_np[:]["frame_index_interval"],
    )
    frame_end = np.max(
        scenes_np[:]["frame_index_interval"],
    )
    frames_np = cast("npt.NDArray", frames[frame_start:frame_end])
    agent_start = np.min(
        frames_np[:]["agent_index_interval"],
    )
    agent_end = np.max(
        frames_np[:]["agent_index_interval"],
    )
    agents_np = cast("npt.NDArray", agents[agent_start:agent_end])
    for scene_data in scenes_np:
        scene = LyftScene(scene_data=scene_data)
        yield _scene_to_pandas(
            scene=scene,
            frames=frames_np,
            agents=agents_np,
            frame_offset=frame_start,
            agent_offset=agent_start,
        )


def _scene_to_pandas(
    scene: LyftScene,
    frames: npt.NDArray,
    agents: npt.NDArray,
    frame_offset: int = 0,
    agent_offset: int = 0,
) -> pd.DataFrame:
    data_rows: list[_Frame] = []
    l_frames = LyftFrames(
        frames=frames,
        interval=scene.frame_interval,
        frame_offset=frame_offset,
    )
    for frame, l_agents in enumerate(
        _agents_iterator(agents, l_frames, agent_offset),
    ):
        ego_row = l_frames.ego_data(frame_index=frame)
        ego_row["scene_name"] = scene.scene_name
        data_rows.append(ego_row)
        for agent in l_agents.agents:
            centroid = agent["centroid"]
            velocity = agent["velocity"]
            data_rows.append(
                _Frame(
                    frame=frame,
                    track_id=int(agent["track_id"]),
                    x=float(centroid[0]),
                    y=float(centroid[1]),
                    vx=float(velocity[0]),
                    vy=float(velocity[1]),
                    yaw=float(agent["yaw"]),
                    category=_get_category_from_label_probabilities(
                        label_probabilities=agent["label_probabilities"],
                    ),
                    scene_name=scene.scene_name,
                ),
            )

    return pd.DataFrame(data_rows)


def _get_category_from_label_probabilities(
    label_probabilities: npt.NDArray,
) -> CategoryStr:
    """Get the category from label probabilities."""
    categories: list[CategoryStr] = [
        "undefined",  # not set
        "undefined",  # unknown
        "undefined",  # dont care
        "car",
        "van",
        "tram",
        "bus",
        "truck",
        "emergency_vehicle",
        "undefined",  # "other_vehicle"
        "bicycle",
        "motorcycle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "animal",
        "undefined",  # dont care
    ]
    max_index = np.argmax(label_probabilities)
    return categories[max_index] if max_index < len(categories) else "undefined"


if __name__ == "__main__":
    # Example usage
    directory = Path("data/sample/sample.zarr")  # 16220

    # Example usage
    scenes = get_lyft_scenes_as_pandas_lazy(directory, start=0, batch_size=1000)
    start_time = time.time()
    count = 0
    for scene_df in scenes:
        count += 1
    print(f"Processed {count} scenes in {time.time() - start_time:.2f} seconds.")
