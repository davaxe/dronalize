from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config import LoaderConfig
from dronalize.datasets.waymo.map.builder import WaymoMapBuilder
from dronalize.datasets.waymo.protos import lean_map_pb2, lean_scenario_pb2
from dronalize.exceptions import SplitNotSupportedError
from dronalize.loading import BaseSceneLoader
from dronalize.loading.loader import IngestOutput, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline
from dronalize.scene import POSITIONS_VELOCITY_YAW_V1

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.config.map import MapConfig
    from dronalize.maps.graph import MapGraph
    from dronalize.maps.resolver import MapResolver
    from dronalize.scene import Scene, SceneSchema


class WaymoLoader(BaseSceneLoader[Path]):
    """Loader for Waymo Open Dataset scenarios stored in TFRecord format."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        """Initialize.

        The WAYMO dataset stores map and scenario data together in TFRecord
        files, in a protobuf format. That is why it is possible to include map
        data in the processing pipeline without a separate map file. However,
        including map decreases the total processing speed, but is faster than
        loading map data separately for each scenario. The map data

        `interp_distance` and `min_distance` parameters control the density of
        map points after processing, and do not do anything if map inclusion is
        disabled.

        Parameters
        ----------
        data_root : Path or str
            Root directory of the Waymo dataset.  This directory should
            contain `training/`, `validation/`, and `testing/`
            subdirectories with TFRecord files.
        loader_config : , optional
            Loader configuration override. If None, the default configuration is used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. Can contain one or more predefined splits,
            or `None` to process all sources.
        map_config : MapConfig, optional
            Map configuration. If None, the default configuration is used.

        """
        super().__init__(loader_config=loader_config, map_config=map_config, splits=splits)
        self._data_root: Path = Path(data_root)
        self._include_map: bool = self.map_config.include_map

    @staticmethod
    def _sources_from_dir(data_dir: Path) -> Iterable[Source[Path]]:
        if not data_dir.is_dir():
            return
        for tfrecord_path in sorted(data_dir.glob("*.tfrecord*")):
            yield Source(identifier=tfrecord_path.stem, inner=tfrecord_path)

    @override
    def sources_for_split(self, split: DatasetSplit) -> Iterable[Source[Path]]:
        if split is DatasetSplit.TRAIN:
            return self._sources_from_dir(self._data_root / "training")
        if split is DatasetSplit.VAL:
            return self._sources_from_dir(self._data_root / "validation")
        if split is DatasetSplit.TEST:
            return self._sources_from_dir(self._data_root / "testing")
        raise SplitNotSupportedError(type(self).__name__, split)

    @override
    def num_sources(self) -> int | None:
        splits = (
            self.splits
            if self.splits is not None
            else [DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST]
        )
        return sum(self._count_sources_for_split(split) for split in splits)

    @override
    def ingest(self, source: Source[Path]) -> Iterable[IngestOutput]:
        for raw_data in _read_tfrecord(source.inner):
            scenario = lean_scenario_pb2.LeanScenario.FromString(raw_data)
            resolver: MapResolver | None
            if self._include_map:

                def _resolver(
                    scene: Scene,
                    _raw_data: bytes = raw_data,
                ) -> MapGraph:
                    _ = scene
                    map_data = lean_map_pb2.LeanMapContainer.FromString(_raw_data)
                    return WaymoMapBuilder.from_proto(map_data.map_features).build(
                        min_distance=self.map_config.min_distance,
                        interp_distance=self.map_config.interp_distance,
                    )

                resolver = _resolver
            else:
                resolver = None

            yield _scenario_to_polars(scenario).lazy(), resolver

    @override
    def pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .compose(
                trajectory_pipeline(
                    self.loader_config,
                    velocity_columns=("vx", "vy"),
                )
            )
            # Shift autonomous vehicle id from -1 to 0
            .then(tr.with_columns(pl.col("id") + 1))
        )

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_VELOCITY_YAW_V1

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(input_len=10, output_len=80, sample_time=0.1).with_filtering(
            require_frames=[9]
        )

    @staticmethod
    def _count_sources(data_dir: Path) -> int:
        if not data_dir.is_dir():
            return 0
        return sum(1 for _ in data_dir.glob("*.tfrecord*"))

    def _count_sources_for_split(self, split: DatasetSplit) -> int:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self._data_root / "training")
        if split is DatasetSplit.VAL:
            return self._count_sources(self._data_root / "validation")
        if split is DatasetSplit.TEST:
            return self._count_sources(self._data_root / "testing")
        raise SplitNotSupportedError(type(self).__name__, split)


def _scenario_to_polars(scenario: lean_scenario_pb2.LeanScenario) -> pl.DataFrame:
    ego_track_index = scenario.sdc_track_index
    cat_map = _OBJECT_TYPE_TO_CATEGORY

    # Pre-fetch lists to avoid dictionary lookups in the inner loop
    l_frame: list[int] = []
    l_tid: list[int] = []
    l_x: list[float] = []
    l_y: list[float] = []
    l_vx: list[float] = []
    l_vy: list[float] = []
    l_yaw: list[float] = []
    l_cat: list[int] = []

    for i, track in enumerate(scenario.tracks):
        # Resolve track constants
        t_id = -1 if i == ego_track_index else track.id
        cat_val = cat_map[track.object_type]
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
    # Read entire file into memory at once
    data = path.read_bytes()
    length_data = len(data)
    offset = 0

    # Pre-bind unpack for slight speedup
    unpack_len = struct.Struct("<Q").unpack_from

    while offset < length_data:
        # TFRecord format:
        # uint64 length
        # uint32 masked_crc32_of_length
        # byte   data[length]
        # uint32 masked_crc32_of_data

        if offset + 8 > length_data:
            break

        record_len = unpack_len(data, offset)[0]
        # Skip: Length(8) + CRC(4) = 12 bytes
        # Data starts
        data_start = offset + 12
        data_end = data_start + record_len

        # Yield the raw data slice (zero-copy if using memoryview, but bytes is fast enough)
        yield data[data_start:data_end]

        # Move offset: Length(8) + CRC(4) + Data(record_len) + CRC(4)
        offset = data_end + 4


# Integer values correspond to waymo.open_dataset.Track.ObjectType enum:
# TYPE_UNSET=0, TYPE_VEHICLE=1, TYPE_PEDESTRIAN=2, TYPE_CYCLIST=3, TYPE_OTHER=4
_OBJECT_TYPE_TO_CATEGORY: dict[int, AgentCategory] = {
    0: AgentCategory.UNKNOWN,  # TYPE_UNSET
    1: AgentCategory.CAR,  # TYPE_VEHICLE
    2: AgentCategory.PEDESTRIAN,  # TYPE_PEDESTRIAN
    3: AgentCategory.BICYCLE,  # TYPE_CYCLIST
    4: AgentCategory.UNKNOWN,  # TYPE_OTHER
}
