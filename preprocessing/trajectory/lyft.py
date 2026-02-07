import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

import numpy as np
import numpy.typing as npt
import polars as pl
import zarr

from preprocessing.trajectory.interface import DataProcessor, Frame
from preprocessing.trajectory.utils import Category


@dataclass
class _Source:
    """Source for the lyft case is the interval of scenes to load from the Zarr dataset."""

    interval: tuple[int, int]


class LyftProcessor(DataProcessor[int, _Source, Frame]):
    """Processor for Lyft Level 5 dataset stored in Zarr format."""

    def __init__(
        self,
        zarr_path: Path | str,
        scene_batch_size: int | None = 1000,
    ) -> None:
        super().__init__(validate_output=True)
        self._zarr_path = Path(zarr_path)
        self._scenes: zarr.Array = zarr.open_array(self._zarr_path / "scenes", mode="r")
        self._frames: zarr.Array = zarr.open_array(self._zarr_path / "frames", mode="r")
        self._agents: zarr.Array = zarr.open_array(self._zarr_path / "agents", mode="r")

        self._total_scenes: int = int(self._scenes.shape[0])
        self._batch_size: int = (
            scene_batch_size if scene_batch_size is not None else self._total_scenes
        )

    @override
    def input_len(self) -> int:
        raise NotImplementedError

    @override
    def output_len(self) -> int:
        raise NotImplementedError

    @override
    def sources(self) -> Iterable[tuple[int, _Source]]:
        current: int = 0
        while current < self._total_scenes:
            end = min(current + self._batch_size, self._total_scenes)
            scene_interval = (current, end)
            yield current, _Source(interval=scene_interval)
            current += self._batch_size

    @override
    def load_raw(self, source: _Source) -> Iterable[pl.DataFrame]:
        scene_interval = source.interval
        if scene_interval is not None:
            start, end = scene_interval
            scenes_np = cast("npt.NDArray", self._scenes[start:end])
        else:
            scenes_np = cast("npt.NDArray", self._scenes[:])

        frame_start = np.min(scenes_np[:]["frame_index_interval"])
        frame_end = np.max(scenes_np[:]["frame_index_interval"])
        frames_np = cast("npt.NDArray", self._frames[frame_start:frame_end])
        agent_start = np.min(frames_np[:]["agent_index_interval"])
        agent_end = np.max(frames_np[:]["agent_index_interval"])
        agents_np = cast("npt.NDArray", self._agents[agent_start:agent_end])

        for scene_data in scenes_np:
            # TODO: This is ~25 seconds sequence; need to split into smaller scene
            # chunks for correct format, i.e.
            yield _scene_to_polars(
                scene=_LyftScene(scene_data=scene_data),
                frames=frames_np,
                agents=agents_np,
                frame_offset=frame_start,
                agent_offset=agent_start,
            )[1]

    @override
    def normalize(self, df: pl.DataFrame) -> pl.DataFrame | None:
        # TODO: Add correct normalization step
        print(df.filter(pl.col("id") == 1))
        print(df.select(pl.col("id").n_unique()))
        return df


@dataclass
class _LyftScene:
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


_CATEGORY_LOOKUP = np.array(
    [
        Category.UNKNOWN.value,  # 0
        Category.UNKNOWN.value,  # 1
        Category.UNKNOWN.value,  # 2
        Category.CAR.value,  # 3
        Category.VAN.value,  # 4
        Category.TRAM.value,  # 5
        Category.BUS.value,  # 6
        Category.TRUCK.value,  # 7
        Category.EMERGENCY_VEHICLE.value,  # 8
        Category.UNKNOWN.value,  # 9
        Category.BICYCLE.value,  # 10
        Category.MOTORCYCLE.value,  # 11
        Category.BICYCLE.value,  # 12
        Category.MOTORCYCLE.value,  # 13
        Category.PEDESTRIAN.value,  # 14
        Category.ANIMAL.value,  # 15
        Category.UNKNOWN.value,  # 16
        Category.UNKNOWN.value,
        Category.UNKNOWN.value,
        Category.UNKNOWN.value,
    ]
)


def _scene_to_polars(
    scene: _LyftScene,
    frames: npt.NDArray,
    agents: npt.NDArray,
    frame_offset: int = 0,
    agent_offset: int = 0,
) -> tuple[str, pl.DataFrame]:
    # This function has been optimized for performance using vectorized operations.
    scene_start, scene_end = scene.frame_interval
    local_f_start = scene_start - frame_offset
    local_f_end = scene_end - frame_offset

    scene_frames = frames[local_f_start:local_f_end]
    n_frames = len(scene_frames)

    ego_ids = np.zeros(n_frames, dtype=np.int32)
    ego_frames = np.arange(n_frames, dtype=np.int32)
    ego_x = scene_frames["ego_translation"][:, 0]
    ego_y = scene_frames["ego_translation"][:, 1]

    # Create Ego DataFrame
    ego_df = pl.DataFrame(
        {
            "frame": ego_frames,
            "id": ego_ids,
            "x": ego_x,
            "y": ego_y,
            "agent_class": np.full(n_frames, Category.CAR.value, dtype=np.int32),
        }
    )

    intervals = scene_frames["agent_index_interval"]
    counts = intervals[:, 1] - intervals[:, 0]
    total_agents = np.sum(counts)

    if total_agents <= 0:
        return scene.scene_name, ego_df

    # Determine the global start/end indices in the 'agents' array
    # The agents for a scene are contiguous.
    # Start is the first agent of the first frame.
    # End is the last agent of the last frame.
    ag_start_global = intervals[0, 0]
    ag_end_global = intervals[-1, 1]

    # Adjust for the offset of the passed 'agents' chunk
    local_a_start = ag_start_global - agent_offset
    local_a_end = ag_end_global - agent_offset

    # Slice the agents array once for the whole scene
    scene_agents = agents[local_a_start:local_a_end]

    # Create an array of frame indices matching the agents
    # e.g., if frame 0 has 2 agents and frame 1 has 3: [0, 0, 1, 1, 1]
    agent_frame_indices = np.repeat(np.arange(n_frames), counts).astype(np.int32)

    agent_ids = scene_agents["track_id"].astype(np.int32)
    agent_x = scene_agents["centroid"][:, 0]
    agent_y = scene_agents["centroid"][:, 1]

    # Extract probabilities matrix
    probs = scene_agents["label_probabilities"]
    # Find index of max probability for every agent at once
    max_indices = np.argmax(probs, axis=1)
    # Map indices to Category values using the lookup table
    safe_indices = np.minimum(max_indices, len(_CATEGORY_LOOKUP) - 1)
    agent_classes = _CATEGORY_LOOKUP[safe_indices].astype(np.int32)

    agent_df = pl.DataFrame(
        {
            "frame": agent_frame_indices,
            "id": agent_ids,
            "x": agent_x,
            "y": agent_y,
            "agent_class": agent_classes,
        }
    )
    return scene.scene_name, pl.concat([ego_df, agent_df])


if __name__ == "__main__":
    # Example usage
    directory = Path("data/sample/sample.zarr")  # 16220

    # Example usage
    scenes = LyftProcessor(directory, scene_batch_size=1000)
    start_time = time.time()
    for scene_df in scenes.process_scenes():
        # save a scene to a CSV file
        print(scene_df.inner)
        break
    print(f"Processed scenes in {time.time() - start_time:.2f} seconds.")
