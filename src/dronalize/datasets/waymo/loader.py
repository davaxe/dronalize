"""Loader implementation for the Waymo Open Dataset."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import POSITIONS_VELOCITY_YAW
from dronalize.datasets.shared import utils
from dronalize.datasets.waymo.maps.builder import WaymoMapBuilder
from dronalize.datasets.waymo.protos import lean_map_pb2, lean_scenario_pb2
from dronalize.processing.loading.base import BaseSceneLoader
from dronalize.processing.loading.loader import LoadedSourceData, MapBinding, Source

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import Scene, TrajectorySchema
    from dronalize.processing.models import LoaderRequest


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


class WaymoLoader(BaseSceneLoader):
    """Loader for Waymo scenarios stored in TFRecord format."""

    def __init__(self, *, data_root: Path | str, request: LoaderRequest) -> None:
        """Initialize the Waymo loader."""
        super().__init__(data_root=data_root, request=request)
        self.root: Path = Path(data_root)
        self._include_map: bool = self.map_config is not None

    @staticmethod
    def _sources_from_dir(data_dir: Path) -> Iterable[Source[Path]]:
        if not data_dir.is_dir():
            return
        for tfrecord_path in sorted(data_dir.glob("*.tfrecord*")):
            yield Source(identifier=tfrecord_path.stem, data=tfrecord_path)

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[Path]]:
        if split is DatasetSplit.TRAIN:
            return self._sources_from_dir(self.root / "training")
        if split is DatasetSplit.VAL:
            return self._sources_from_dir(self.root / "validation")
        return self._sources_from_dir(self.root / "testing")

    @override
    def num_sources(self) -> int | None:
        return sum(
            self._count_sources_for_split(split) for split in self.native_splits or _NATIVE_SPLITS
        )

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        for scenario_index, raw_data in enumerate(_read_tfrecord(source.data)):
            scenario = lean_scenario_pb2.LeanScenario.FromString(raw_data)
            yield LoadedSourceData(
                frame=_scenario_to_polars(scenario).lazy().with_columns(pl.col("id").add(1)),
                map_binding=MapBinding(
                    map_key=f"{source.identifier}:{scenario_index}",
                    metadata={"raw_map": raw_data} if self._include_map else {},
                ),
            )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_YAW

    @override
    def resolve_map(self, scene: Scene, map_binding: MapBinding | None = None) -> MapGraph | None:
        if not self._include_map or map_binding is None or self.map_config is None:
            return None
        raw_map = map_binding.metadata.get("raw_map")
        if not isinstance(raw_map, bytes):
            return None
        map_data = lean_map_pb2.LeanMapContainer.FromString(raw_map)
        map_config = self.map_config
        map_graph = WaymoMapBuilder.from_proto(map_data.map_features).build(
            min_distance=map_config.min_distance,
            interp_distance=map_config.interp_distance,
        )
        return utils.extract_based_on_scene(map_graph, scene, map_config.extraction)

    @staticmethod
    def _count_sources(data_dir: Path) -> int:
        return sum(1 for _ in data_dir.glob("*.tfrecord*")) if data_dir.is_dir() else 0

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self.root / "training")
        if split is DatasetSplit.VAL:
            return self._count_sources(self.root / "validation")
        return self._count_sources(self.root / "testing")


def _scenario_to_polars(scenario: lean_scenario_pb2.LeanScenario) -> pl.DataFrame:
    ego_track_index = scenario.sdc_track_index
    l_frame: list[int] = []
    l_tid: list[int] = []
    l_x: list[float] = []
    l_y: list[float] = []
    l_vx: list[float] = []
    l_vy: list[float] = []
    l_yaw: list[float] = []
    l_cat: list[int] = []

    for i, track in enumerate(scenario.tracks):
        t_id = -1 if i == ego_track_index else track.id
        cat_val = _OBJECT_TYPE_TO_CATEGORY[track.object_type]
        for frame_idx, state in enumerate(track.states):
            if not state.valid:
                continue
            l_frame.append(frame_idx)
            l_tid.append(t_id)
            l_x.append(state.center_x)
            l_y.append(state.center_y)
            l_vx.append(state.velocity_x)
            l_vy.append(state.velocity_y)
            l_yaw.append(state.heading)
            l_cat.append(cat_val)

    return pl.DataFrame(
        {
            "frame": l_frame,
            "id": l_tid,
            "x": l_x,
            "y": l_y,
            "vx": l_vx,
            "vy": l_vy,
            "yaw": l_yaw,
            "agent_category": l_cat,
        },
        schema={
            "frame": pl.Int32,
            "id": pl.Int32,
            "x": pl.Float64,
            "y": pl.Float64,
            "vx": pl.Float64,
            "vy": pl.Float64,
            "yaw": pl.Float64,
            "agent_category": pl.Int32,
        },
    )


def _read_tfrecord(path: Path) -> Iterable[bytes]:
    data = path.read_bytes()
    offset = 0
    unpack_len = struct.Struct("<Q").unpack_from
    while offset < len(data):
        if offset + 8 > len(data):
            break
        record_len = unpack_len(data, offset)[0]
        data_start = offset + 12
        data_end = data_start + record_len
        yield data[data_start:data_end]
        offset = data_end + 4


_OBJECT_TYPE_TO_CATEGORY: dict[int, AgentCategory] = {
    0: AgentCategory.UNKNOWN,
    1: AgentCategory.CAR,
    2: AgentCategory.PEDESTRIAN,
    3: AgentCategory.BICYCLE,
    4: AgentCategory.UNKNOWN,
}
