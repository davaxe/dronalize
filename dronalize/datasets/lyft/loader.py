from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from typing_extensions import override
from zarr.creation import open_array

import dronalize.pipeline.transforms as tr
from dronalize.core.datatypes.categories import AgentCategory
from dronalize.core.datatypes.loader_config import LoaderConfig
from dronalize.core.protocols.loader import BaseSceneLoader, IngestOutput, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zarr.core import Array


@dataclass
class _Source:
    """Source for the lyft case is the interval of scenes to load from the Zarr dataset."""

    interval: tuple[int, int]


class LyftLoader(BaseSceneLoader[int, _Source]):
    """Processor for Lyft Level 5 dataset stored in Zarr format."""

    def __init__(
        self,
        data_dir: Path | str,
        loader_config: LoaderConfig | None = None,
        *,
        scene_batch_size: int | None = 1000,
    ) -> None:
        """Initialize the processor.

        Parameters
        ----------
        data_dir : Path or str
            Path to the Zarr dataset directory.
        scene_batch_size : int, optional
            Number of scenes to load in each batch. Higher batch size may lead
            to faster processing at diminishing returns, but also higher memory
            usage. `None` is not recommended for large amounts of data.
        loader_config : LoaderConfig, optional
            Processor configuration override. If None, the default
            configuration will be used.

        """
        super().__init__(loader_config=loader_config, enforce_schema=True)
        self._zarr_path = Path(data_dir)
        self._scenes: Array = open_array(self._zarr_path / "scenes", mode="r")
        self._frames: Array = open_array(self._zarr_path / "frames", mode="r")
        self._agents: Array = open_array(self._zarr_path / "agents", mode="r")

        self._total_scenes: int = int(self._scenes.shape[0])
        self._batch_size: int = (
            scene_batch_size if scene_batch_size is not None else self._total_scenes
        )

    @override
    def all_sources(self) -> Iterable[Source[int, _Source]]:
        current: int = 0
        while current < self._total_scenes:
            end = min(current + self._batch_size, self._total_scenes)
            yield Source(
                identifier=current,
                inner=_Source(interval=(current, end)),
            )
            current += self._batch_size

    @override
    def num_sources(self) -> int | None:
        return (self._total_scenes + self._batch_size - 1) // self._batch_size

    @override
    def ingest(self, source: Source[int, _Source]) -> Iterable[IngestOutput]:
        start, end = source.inner.interval
        scenes_np = cast("npt.NDArray", self._scenes[start:end])

        frame_start = np.min(scenes_np[:]["frame_index_interval"])
        frame_end = np.max(scenes_np[:]["frame_index_interval"])
        frames_np = cast("npt.NDArray", self._frames[frame_start:frame_end])
        agent_start = np.min(frames_np[:]["agent_index_interval"])
        agent_end = np.max(frames_np[:]["agent_index_interval"])
        agents_np = cast("npt.NDArray", self._agents[agent_start:agent_end])

        for scene_data in scenes_np:
            df = _scene_to_polars(
                scene_data,
                frames=frames_np,
                agents=agents_np,
                frame_offset=frame_start,
                agent_offset=agent_start,
            )
            yield df.lazy(), None

    @override
    def pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .compose(
                trajectory_pipeline(self.loader_config, derivative_rename=self.derivative_names())
            )
            .then(tr.yaw_from_vel())
        )

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(
                input_len=20,
                output_len=50,
                sample_time=0.1,
            )
            .with_window(step_size=20)
            .with_filtering(
                min_agents=1,
                filter_agent_category={AgentCategory.UNIMPORTANT},
                require_frames=[19, -1],
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


_CATEGORY_LOOKUP = np.array([
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
])


def _scene_to_polars(
    scene_data: npt.NDArray,
    frames: npt.NDArray,
    agents: npt.NDArray,
    frame_offset: int = 0,
    agent_offset: int = 0,
) -> pl.DataFrame:
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
    ego_df = pl.DataFrame({
        "frame": ego_frames,
        "id": ego_ids,
        "x": ego_x,
        "y": ego_y,
        "agent_category": np.full(n_frames, AgentCategory.CAR.value, dtype=np.int32),
    })

    intervals = scene_frames["agent_index_interval"]
    counts = intervals[:, 1] - intervals[:, 0]
    total_agents = np.sum(counts)

    if total_agents <= 0:
        return ego_df

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

    agent_df = pl.DataFrame({
        "frame": agent_frame_indices,
        "id": agent_ids,
        "x": agent_x,
        "y": agent_y,
        "agent_category": agent_categories,
    })
    return pl.concat([ego_df, agent_df])
