"""Loader implementation for the Lyft Level 5 dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import Field
from typing_extensions import override
from zarr.creation import open_array  # pyright: ignore[reportUnknownVariableType]

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.errors import SplitNotSupportedError
from dronalize.core.scene import POSITIONS_ONLY
from dronalize.datasets.shared import utils
from dronalize.processing.loading.base import (
    BaseSceneLoader,
    LoaderOptions,
    LoaderSplitCapabilities,
)
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.processing.maps.resolver import no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from zarr import Array

    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.maps.resolver import MapResolver
    from dronalize.processing.models import LoaderRequest


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL)


class LyftLoaderOptions(LoaderOptions):
    """Dataset-owned config for the Lyft loader."""

    scene_batch_size: int = Field(default=100, ge=1)


@dataclass
class _Source:
    interval: tuple[int, int]
    split: DatasetSplit


@dataclass(slots=True)
class _ArrayData:
    scenes: Array
    frames: Array
    agents: Array

    @property
    def total_scenes(self) -> int:
        return self.scenes.shape[0]


class LyftLoader(BaseSceneLoader[_Source, LyftLoaderOptions]):
    """Loader for Lyft Level 5 scenes stored in Zarr format."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_scene_split=True
    )

    def __init__(
        self,
        *,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        """Initialize the Lyft loader."""
        super().__init__(data_root=data_root, request=request, resources=resources)
        self._data: dict[DatasetSplit, _ArrayData] = {}

    @classmethod
    @override
    def loader_options_model(cls) -> type[LyftLoaderOptions]:
        return LyftLoaderOptions

    def _get_arrays(self, split: DatasetSplit) -> _ArrayData:
        if split not in self._data:
            split_paths = {
                DatasetSplit.VAL: "validate/validate.zarr",
                DatasetSplit.TRAIN: "train/train.zarr",
            }
            if split not in split_paths:
                raise SplitNotSupportedError(type(self).__name__, split)

            split_path = self.root / split_paths[split]
            self._data[split] = _ArrayData(
                scenes=open_array(split_path / "scenes", mode="r"),
                frames=open_array(split_path / "frames", mode="r"),
                agents=open_array(split_path / "agents", mode="r"),
            )
        return self._data[split]

    def _generate_sources(self, split: DatasetSplit) -> Iterable[Source[_Source]]:
        arrays = self._get_arrays(split)
        current: int = 0
        while current < arrays.total_scenes:
            end = min(current + self.loader_options.scene_batch_size, arrays.total_scenes)
            yield Source(current, data=_Source(interval=(current, end), split=split))
            current += self.loader_options.scene_batch_size

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[_Source]]:
        if split in {DatasetSplit.TRAIN, DatasetSplit.VAL}:
            yield from self._generate_sources(split)
            return
        raise SplitNotSupportedError(type(self).__name__, split)

    @override
    def num_sources(self) -> int | None:
        return sum(
            self._count_sources_for_split(split) for split in self.native_splits or _NATIVE_SPLITS
        )

    @staticmethod
    def _source_count(total_scenes: int, batch_size: int) -> int:
        return (total_scenes + batch_size - 1) // batch_size

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        return self._source_count(
            self._get_arrays(split).total_scenes, self.loader_options.scene_batch_size
        )

    @override
    def load_source(self, source: Source[_Source]) -> Iterable[LoadedSourceData]:
        start, end = source.data.interval
        arrays = self._get_arrays(source.data.split)
        scenes_np = cast("npt.NDArray[np.void]", arrays.scenes[start:end])

        frame_start = np.min(scenes_np[:]["frame_index_interval"])
        frame_end = np.max(scenes_np[:]["frame_index_interval"])
        frames_np = cast("npt.NDArray[np.void]", arrays.frames[frame_start:frame_end])
        agent_start = np.min(frames_np[:]["agent_index_interval"])
        agent_end = np.max(frames_np[:]["agent_index_interval"])
        agents_np = cast("npt.NDArray[np.void]", arrays.agents[agent_start:agent_end])

        for scene_data in scenes_np:
            yield LoadedSourceData(
                frame=_scene_to_polars(
                    scene_data,
                    frames=frames_np,
                    agents=agents_np,
                    frame_offset=frame_start,
                    agent_offset=agent_start,
                ).lazy()
            )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY

    @override
    def map_resolver(self) -> MapResolver:
        shared_maps = self.resources.shared_maps
        if not isinstance(shared_maps, str) or self.map_config is None:
            return no_map()
        return shared_map(shared_maps, utils.extract_fn(self.map_config.extraction))


@dataclass
class _LyftScene:
    scene_data: npt.NDArray[np.void]

    @property
    def frame_interval(self) -> tuple[int, int]:
        start, end = self.scene_data["frame_index_interval"]
        return int(start), int(end)


_CATEGORY_LOOKUP = np.array([
    AgentCategory.UNIMPORTANT.value,
    AgentCategory.UNKNOWN.value,
    AgentCategory.UNIMPORTANT.value,
    AgentCategory.CAR.value,
    AgentCategory.VAN.value,
    AgentCategory.TRAM.value,
    AgentCategory.BUS.value,
    AgentCategory.TRUCK.value,
    AgentCategory.EMERGENCY_VEHICLE.value,
    AgentCategory.UNKNOWN.value,
    AgentCategory.BICYCLE.value,
    AgentCategory.MOTORCYCLE.value,
    AgentCategory.BICYCLE.value,
    AgentCategory.MOTORCYCLE.value,
    AgentCategory.PEDESTRIAN.value,
    AgentCategory.ANIMAL.value,
    AgentCategory.UNIMPORTANT.value,
])


def _scene_to_polars(
    scene_data: npt.NDArray[np.void],
    frames: npt.NDArray[np.void],
    agents: npt.NDArray[np.void],
    frame_offset: int = 0,
    agent_offset: int = 0,
) -> pl.DataFrame:
    scene = _LyftScene(scene_data=scene_data)
    scene_start, scene_end = scene.frame_interval
    scene_frames = frames[scene_start - frame_offset : scene_end - frame_offset]
    n_frames = len(scene_frames)

    ego_df = pl.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int32),
        "id": np.zeros(n_frames, dtype=np.int32),
        "x": scene_frames["ego_translation"][:, 0],
        "y": scene_frames["ego_translation"][:, 1],
        "agent_category": np.full(n_frames, AgentCategory.CAR.value, dtype=np.int32),
    })

    intervals = scene_frames["agent_index_interval"]
    counts = intervals[:, 1] - intervals[:, 0]
    if np.sum(counts) <= 0:
        return ego_df

    scene_agents = agents[intervals[0, 0] - agent_offset : intervals[-1, 1] - agent_offset]
    safe_indices = np.minimum(
        np.argmax(scene_agents["label_probabilities"], axis=1), len(_CATEGORY_LOOKUP) - 1
    )
    agent_df = pl.DataFrame({
        "frame": np.repeat(np.arange(n_frames), counts).astype(np.int32),
        "id": scene_agents["track_id"].astype(np.int32),
        "x": scene_agents["centroid"][:, 0],
        "y": scene_agents["centroid"][:, 1],
        "agent_category": _CATEGORY_LOOKUP[safe_indices].astype(np.int32),
    })
    return pl.concat([ego_df, agent_df])
