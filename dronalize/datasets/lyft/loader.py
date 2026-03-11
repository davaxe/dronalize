from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from typing_extensions import override
from zarr.creation import open_array

import dronalize.pipeline.transforms as tr
from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config.loader import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.common import utils
from dronalize.loading import BaseSceneLoader, IngestOutput, Source
from dronalize.maps.resolver import no_map, shared_map
from dronalize.pipeline import Pipeline
from dronalize.pipeline.factories import trajectory_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from zarr.core import Array

    from dronalize.maps import MapResolver


@dataclass
class _Source:
    """Source for the lyft case is the interval of scenes to load from the Zarr dataset."""

    interval: tuple[int, int]
    split: DatasetSplit


@dataclass(slots=True)
class _ArrayData:
    """Helper class to hold the Zarr arrays for a dataset split."""

    scenes: Array
    frames: Array
    agents: Array

    @property
    def total_scenes(self) -> int:
        """Return the total number of scenes in this split."""
        return self.scenes.shape[0]


class LyftLoader(BaseSceneLoader[_Source]):
    """Loader for Lyft Level 5 scenes stored in Zarr format."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        *,
        scene_batch_size: int | None = 100,
    ) -> None:
        """Initialize the dataset loader.

        Parameters
        ----------
        data_root : Path or str
            Root directory of the dataset.
        scene_batch_size : int, optional
            Number of scenes to load in each batch. Higher batch size may lead
            to faster processing at diminishing returns, but also higher memory
            usage. `None` is not recommended for large amounts of data.
        loader_config : LoaderConfig, optional
            Loader configuration override. If None, the default configuration
            is used.
        """
        requested_splits = (
            [DatasetSplit.ALL]
            if splits is None
            else [splits]
            if isinstance(splits, DatasetSplit)
            else list(splits)
        )
        if DatasetSplit.TEST in requested_splits:
            msg = "does not support splits containing TEST."
            raise self._invalid_loader_argument(msg)
        if splits is not None and not isinstance(splits, DatasetSplit):
            splits = requested_splits

        super().__init__(loader_config=loader_config, map_config=map_config, splits=splits)
        self._data_root: Path = self._normalize_data_root(data_root)
        self._data: dict[DatasetSplit, _ArrayData] = {}
        self._batch_size: int | None = scene_batch_size

    def _get_arrays(self, split: DatasetSplit) -> _ArrayData:
        """Lazily load and return the arrays for a given split."""
        if split not in self._data:
            split_paths = {
                DatasetSplit.VAL: "validate/validate.zarr",
                DatasetSplit.TRAIN: "train/train.zarr",
            }

            if split not in split_paths:
                msg = f"does not support split={split.name}."
                raise self._invalid_loader_argument(msg)

            split_path = self._data_root / split_paths[split]
            scenes = open_array(split_path / "scenes", mode="r")
            frames = open_array(split_path / "frames", mode="r")
            agents = open_array(split_path / "agents", mode="r")
            self._data[split] = _ArrayData(scenes=scenes, frames=frames, agents=agents)

        return self._data[split]

    def _generate_sources(self, split: DatasetSplit) -> Iterable[Source[_Source]]:
        arrays = self._get_arrays(split)
        total_scenes = arrays.total_scenes
        batch_size = self._batch_size or total_scenes

        current: int = 0
        while current < total_scenes:
            end = min(current + batch_size, total_scenes)
            yield Source(
                current,
                inner=_Source(interval=(current, end), split=split),
            )
            current += batch_size

    @override
    def all_sources(self) -> Iterable[Source[_Source]]:
        yield from self.train_sources()
        yield from self.validate_sources()

    @override
    def train_sources(self) -> Iterable[Source[_Source]]:
        yield from self._generate_sources(DatasetSplit.TRAIN)

    @override
    def validate_sources(self) -> Iterable[Source[_Source]]:
        yield from self._generate_sources(DatasetSplit.VAL)

    @override
    def num_sources(self) -> int | None:
        def _count_sources(split: DatasetSplit) -> int:
            total_scenes = self._get_arrays(split).total_scenes
            return self._source_count(total_scenes, self._batch_size)

        total = 0
        for split in self._splits:
            if split is DatasetSplit.TRAIN:
                total += _count_sources(DatasetSplit.TRAIN)
            elif split is DatasetSplit.VAL:
                total += _count_sources(DatasetSplit.VAL)
            else:
                total += _count_sources(DatasetSplit.TRAIN)
                total += _count_sources(DatasetSplit.VAL)
        return total

    @staticmethod
    def _source_count(total_scenes: int, batch_size: int | None) -> int:
        if batch_size is None:
            return 1
        return (total_scenes + batch_size - 1) // batch_size

    @override
    def ingest(self, source: Source[_Source]) -> Iterable[IngestOutput]:  # type: ignore[override]
        start, end = source.inner.interval

        # Now uses the lazy loader directly to ensure availability
        arrays = self._get_arrays(source.inner.split)

        scenes_np = cast("npt.NDArray[np.void]", arrays.scenes[start:end])

        frame_start = np.min(scenes_np[:]["frame_index_interval"])
        frame_end = np.max(scenes_np[:]["frame_index_interval"])
        frames_np = cast("npt.NDArray[np.void]", arrays.frames[frame_start:frame_end])

        agent_start = np.min(frames_np[:]["agent_index_interval"])
        agent_end = np.max(frames_np[:]["agent_index_interval"])
        agents_np = cast("npt.NDArray[np.void]", arrays.agents[agent_start:agent_end])

        for scene_data in scenes_np:
            df = _scene_to_polars(
                scene_data,
                frames=frames_np,
                agents=agents_np,
                frame_offset=frame_start,
                agent_offset=agent_start,
            )
            yield df.lazy(), self.map_resolver()

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

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.auto_extraction()

    @override
    def map_resolver(self) -> MapResolver:
        if self._shared_memory_name is None:
            return no_map()
        return shared_map(self._shared_memory_name, utils.extract_fn(self.map_config.extraction))


@dataclass
class _LyftScene:
    """A single scene in the Lyft dataset."""

    scene_data: npt.NDArray[np.void]

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
    scene_data: npt.NDArray[np.void],
    frames: npt.NDArray[np.void],
    agents: npt.NDArray[np.void],
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
