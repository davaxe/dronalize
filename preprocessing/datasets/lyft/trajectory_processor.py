from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import numpy as np
import numpy.typing as npt
import polars as pl
from zarr.creation import open_array

from preprocessing.common.trajectory_utils import (
    filter_scene_expr,
    resample_tracks,
    sliding_window,
    yaw_from_vel,
)
from preprocessing.core.categories import AgentCategory
from preprocessing.core.interface import (
    DataProcessor,
    ProcessorConfig,
    Resampling,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zarr.core import Array


@dataclass
class _Source:
    """Source for the lyft case is the interval of scenes to load from the Zarr dataset."""

    interval: tuple[int, int]


class LyftProcessor(DataProcessor[int, _Source]):
    """Processor for Lyft Level 5 dataset stored in Zarr format."""

    def __init__(
        self,
        zarr_path: Path | str,
        scene_batch_size: int | None = 1000,
        config: ProcessorConfig | None = None,
    ) -> None:
        if config is None:
            config = self._default_config()

        super().__init__(processor_config=config, enforce_schema=True)
        self._zarr_path = Path(zarr_path)
        self._scenes: Array = open_array(self._zarr_path / "scenes", mode="r")
        self._frames: Array = open_array(self._zarr_path / "frames", mode="r")
        self._agents: Array = open_array(self._zarr_path / "agents", mode="r")

        self._total_scenes: int = int(self._scenes.shape[0])
        self._batch_size: int = (
            scene_batch_size if scene_batch_size is not None else self._total_scenes
        )

    @override
    def sources(self) -> Iterable[tuple[int, _Source]]:
        current: int = 0
        while current < self._total_scenes:
            end = min(current + self._batch_size, self._total_scenes)
            scene_interval = (current, end)
            yield current, _Source(interval=scene_interval)
            current += self._batch_size

    @override
    def load_raw(self, source: _Source) -> Iterable[pl.LazyFrame]:
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
        resampling = self.processor_config.resampling or Resampling(1, 1)

        for scene_data in scenes_np:
            scenes = _scene_to_polars(
                scene_data,
                frames=frames_np,
                agents=agents_np,
                frame_offset=frame_start,
                agent_offset=agent_start,
            )[1].lazy()

            group_by: list[str] = []
            if self.processor_config.window_params is not None:
                scenes = sliding_window(
                    scenes,
                    window_size=self.sequence_length,
                    step_size=self.processor_config.window_params.step_size,
                    sliding_col="frame",
                    is_sorted=False,
                    return_iterable=False,
                )
                group_by.append("window_index")

            source_filtered = scenes.filter(
                filter_scene_expr(
                    self.processor_config,
                    group_by=group_by[-1] if len(group_by) > 0 else None,
                    category_column="agent_category",
                )
            )
            group_by.append("id")
            source_filtered = source_filtered.filter(pl.len().over(group_by) >= 2)

            processed_source = resample_tracks(
                source_filtered,
                resampling.up,
                resampling.down,
                group_by=group_by,
                add_derivative=True,
                add_second_derivative=True,
                method=resampling.method,
                dt=self.processor_config.sample_time,
                derivative_rename=self.derivative_names(),
                forward_fill=["agent_category"],
            )

            if self.processor_config.window_params is None:
                yield processed_source.lazy()
                return

            for _, group in processed_source.collect().group_by("window_index"):
                yield group.lazy().drop("window_index")

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return yaw_from_vel(df, yaw_col="yaw")

    @staticmethod
    def _default_config() -> ProcessorConfig:
        return (
            ProcessorConfig(
                input_len=20,
                output_len=50,
                sample_time=0.1,
            )
            .window_parameters(step_size=20)
            .scene_filtering_parameters(
                min_agents=1,
                require_prediction_frame=True,
                filter_agent_category={AgentCategory.UNIMPORTANT},
                require_frames=[-1],
            )
        )


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
        AgentCategory.UNIMPORTANT.value,  # 0 (not set)
        AgentCategory.UNKNOWN.value,  # 1 (unknown)
        AgentCategory.UNIMPORTANT.value,  # 2 (don't care)
        AgentCategory.CAR.value,  # 3
        AgentCategory.VAN.value,  # 4
        AgentCategory.TRAM.value,  # 5
        AgentCategory.BUS.value,  # 6
        AgentCategory.TRUCK.value,  # 7
        AgentCategory.EMERGENCY_VEHICLE.value,  # 8
        AgentCategory.UNKNOWN.value,  # 9 ("other_vehicle")
        AgentCategory.BICYCLE.value,  # 10
        AgentCategory.MOTORCYCLE.value,  # 11
        AgentCategory.BICYCLE.value,  # 12
        AgentCategory.MOTORCYCLE.value,  # 13
        AgentCategory.PEDESTRIAN.value,  # 14
        AgentCategory.ANIMAL.value,  # 15
        AgentCategory.UNIMPORTANT.value,  # 16 (don't care)
    ]
)


def _scene_to_polars(
    scene_data: npt.NDArray,
    frames: npt.NDArray,
    agents: npt.NDArray,
    frame_offset: int = 0,
    agent_offset: int = 0,
) -> tuple[str, pl.DataFrame]:
    # This function has been optimized for performance using vectorized operations.
    scene = _LyftScene(scene_data=scene_data)
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
            "agent_category": np.full(
                n_frames, AgentCategory.CAR.value, dtype=np.int32
            ),
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
    agent_categories = _CATEGORY_LOOKUP[safe_indices].astype(np.int32)

    agent_df = pl.DataFrame(
        {
            "frame": agent_frame_indices,
            "id": agent_ids,
            "x": agent_x,
            "y": agent_y,
            "agent_category": agent_categories,
        }
    )
    return scene.scene_name, pl.concat([ego_df, agent_df])


def main():
    import time

    """Test."""
    start_time = time.time()
    directory = Path(
        "/home/west/Developer/behavior-prediction/datasets/lyft/validate/validate.zarr"
    )
    directory = Path("data/sample/sample.zarr")
    processor = LyftProcessor(directory, 100)
    _total_scenes = 194608
    count = 0
    for _scene in processor.scenes_iter():
        if count % 1000 == 0:
            print(
                f"Processed {count} scenes in {time.time() - start_time:.2f} seconds."
            )
        count += 1
    print(f"Processed {count} scenes in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
