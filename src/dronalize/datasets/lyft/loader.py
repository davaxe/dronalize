"""Loader implementation for the Lyft Level 5 dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import Field
from typing_extensions import override
from zarr.creation import open_array

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.errors import SplitNotSupportedError
from dronalize.core.scene import POSITIONS_ONLY
from dronalize.datasets.shared import utils
from dronalize.processing.filters import Filter
from dronalize.processing.filters.agent import RequireFrames
from dronalize.processing.filters.cleanup import ExcludeCategories
from dronalize.processing.filters.scene import MinimumAgents
from dronalize.processing.ingest.base import BaseSceneLoader, LoaderOptions, LoaderSplitCapabilities
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.ingest.loader import IngestedData, MapBinding, Source
from dronalize.processing.maps.config import MapConfig
from dronalize.processing.maps.resolver import no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zarr import Array

    from dronalize.core.scene import SceneSchema
    from dronalize.processing.ingest.splits import SplitConfig
    from dronalize.processing.maps.resolver import MapResolver


class LyftLoaderOptions(LoaderOptions):
    """Loader options for the Lyft Level 5 dataset."""

    scene_batch_size: int = Field(default=100, ge=1)


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


class LyftLoader(BaseSceneLoader[_Source, LyftLoaderOptions]):
    """Loader for Lyft Level 5 scenes stored in Zarr format."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_scene_split=True
    )

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitConfig | None = None,
        *,
        loader_options: LyftLoaderOptions | None = None,
    ) -> None:
        """Initialize the Lyft Level 5 loader.

        Parameters
        ----------
        data_root : Path or str
            Root directory of the extracted Lyft Level 5 dataset.
        loader_options : LyftLoaderOptions, optional
            Dataset-specific loader options. `scene_batch_size` controls how
            many scenes are loaded in each batch.
        loader_config : LoaderConfig, optional
            Loader configuration override. If None, the default configuration
            is used.
        """
        super().__init__(
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
            loader_options=loader_options,
        )
        self._data_root: Path = Path(data_root)
        self._data: dict[DatasetSplit, _ArrayData] = {}
        self._batch_size: int = self.loader_options.scene_batch_size

    @classmethod
    @override
    def loader_options_model(cls) -> type[LyftLoaderOptions]:
        return LyftLoaderOptions

    @classmethod
    @override
    def predefined_splits(cls) -> tuple[DatasetSplit, ...]:
        return (DatasetSplit.TRAIN, DatasetSplit.VAL)

    def _get_arrays(self, split: DatasetSplit) -> _ArrayData:
        """Lazily load and return the arrays for a given split."""
        if split not in self._data:
            split_paths = {
                DatasetSplit.VAL: "validate/validate.zarr",
                DatasetSplit.TRAIN: "train/train.zarr",
            }

            if split not in split_paths:
                raise SplitNotSupportedError(type(self).__name__, split)

            split_path = self._data_root / split_paths[split]
            scenes = open_array(split_path / "scenes", mode="r")
            frames = open_array(split_path / "frames", mode="r")
            agents = open_array(split_path / "agents", mode="r")
            self._data[split] = _ArrayData(scenes=scenes, frames=frames, agents=agents)

        return self._data[split]

    def _generate_sources(self, split: DatasetSplit) -> Iterable[Source[_Source]]:
        arrays = self._get_arrays(split)
        total_scenes = arrays.total_scenes

        current: int = 0
        while current < total_scenes:
            end = min(current + self._batch_size, total_scenes)
            yield Source(current, data=_Source(interval=(current, end), split=split))
            current += self._batch_size

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[_Source]]:
        if split in {DatasetSplit.TRAIN, DatasetSplit.VAL}:
            yield from self._generate_sources(split)
            return
        raise SplitNotSupportedError(type(self).__name__, split)

    @override
    def num_sources(self) -> int | None:
        splits = self.splits if self.splits is not None else self.predefined_splits()
        return sum(self._count_sources_for_split(split) for split in splits)

    @staticmethod
    def _source_count(total_scenes: int, batch_size: int) -> int:
        return (total_scenes + batch_size - 1) // batch_size

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        total_scenes = self._get_arrays(split).total_scenes
        return self._source_count(total_scenes, self._batch_size)

    @override
    def ingest(self, source: Source[_Source]) -> Iterable[IngestedData]:
        start, end = source.data.interval

        # Now uses the lazy loader directly to ensure availability
        arrays = self._get_arrays(source.data.split)
        scenes_np: npt.NDArray[np.void] = cast("npt.NDArray[np.void]", arrays.scenes[start:end])

        frame_start = np.min(scenes_np[:]["frame_index_interval"])
        frame_end = np.max(scenes_np[:]["frame_index_interval"])

        frames_np: npt.NDArray[np.void] = cast(
            "npt.NDArray[np.void]", arrays.frames[frame_start:frame_end]
        )
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
            yield IngestedData(
                frame=df.lazy(), map_binding=MapBinding(map_resolver=self.map_resolver())
            )

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_ONLY

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=20, output_len=50, sample_time=0.1)
            .with_window(step=20)
            .with_filter(
                Filter.define(
                    cleanup_rules=[
                        ExcludeCategories.define(categories=[AgentCategory.UNIMPORTANT])
                    ],
                    scene_rules=[MinimumAgents(minimum=1)],
                    agent_rules=[RequireFrames.define(frames=[19])],
                )
            )
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.relevant_area_extraction()

    @override
    def map_resolver(self) -> MapResolver:
        if self._shared_memory_name is None or self.map_config is None:
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


if __name__ == "__main__":
    from dronalize.datasets.lyft import DESCRIPTOR
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env

    root = resolve_dataset_root_from_env("lyft")
    _ = debug_descriptor(DESCRIPTOR, root)
