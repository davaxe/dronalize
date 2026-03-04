from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.core import AgentCategory, BaseSceneLoader, LoaderConfig
from dronalize.core.datatypes.map_resolver import MapKey, MapResolver, no_map
from dronalize.core.datatypes.split import DatasetSplit
from dronalize.core.protocols.loader import IngestOutput, Source
from dronalize.datasets.waymo.map.graph_builder import WaymoMapGraphBuilder
from dronalize.datasets.waymo.protos import lean_map_pb2, lean_scenario_pb2
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.datatypes.map_graph import MapGraph
    from dronalize.datasets.waymo.protos.lean_map_pb2 import LeanMapContainer


class WaymoLoader(BaseSceneLoader[str, Path]):
    """Processor for Waymo Open Dataset scenarios stored in TFRecord format."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        *,
        split: DatasetSplit = DatasetSplit.ALL,
        include_map: bool = True,
        interp_distance: float | None = None,
        min_distance: float | None = 1.5,
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
        loader_config : LoaderConfig, optional
            Configuration override. If None, the default configuration will be used.
        split : DatasetSplit, optional
            Which dataset split to load.  Defaults to `DatasetSplit.ALL`.
        include_map : bool, optional
            Whether to include map data in the scene. Defaults to True.
        interp_distance : float, optional
            Distance threshold for interpolating map points.
            Defaults to None (no interpolation).
        min_distance : float, optional
            Minimum distance between map points after processing.
            Defaults to 1.5 meters.

        """
        super().__init__(loader_config=loader_config, enforce_schema=True, split=split)
        self._data_root: Path = Path(data_root) if isinstance(data_root, str) else data_root
        self._include_map: bool = include_map
        self._interp_distance: float | None = interp_distance
        self._min_distance: float | None = min_distance

    # ------------------------------------------------------------------
    # Split-aware source discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _sources_from_dir(data_dir: Path) -> Iterable[Source[str, Path]]:
        if not data_dir.is_dir():
            return
        for tfrecord_path in sorted(data_dir.glob("*.tfrecord*")):
            yield Source(identifier=tfrecord_path.stem, inner=tfrecord_path)

    @override
    def all_sources(self) -> Iterable[Source[str, Path]]:
        yield from self.train_sources()
        yield from self.validate_sources()
        yield from self.test_sources()

    @override
    def train_sources(self) -> Iterable[Source[str, Path]]:
        return self._sources_from_dir(self._data_root / "training")

    @override
    def validate_sources(self) -> Iterable[Source[str, Path]]:
        return self._sources_from_dir(self._data_root / "validation")

    @override
    def test_sources(self) -> Iterable[Source[str, Path]]:
        return self._sources_from_dir(self._data_root / "testing")

    # ------------------------------------------------------------------
    # Ingestion / pipeline
    # ------------------------------------------------------------------

    @override
    def num_sources(self) -> int | None:
        dirs: list[Path] = []
        split = self._split
        if split in {DatasetSplit.ALL, DatasetSplit.TRAIN}:
            dirs.append(self._data_root / "training")
        if split in {DatasetSplit.ALL, DatasetSplit.VAL}:
            dirs.append(self._data_root / "validation")
        if split in {DatasetSplit.ALL, DatasetSplit.TEST}:
            dirs.append(self._data_root / "testing")

        return sum(sum(1 for _ in d.glob("*.tfrecord*")) for d in dirs if d.is_dir())

    @override
    def ingest(self, source: Source[str, Path]) -> Iterable[IngestOutput]:
        for raw_data in _read_tfrecord(source.inner):
            scenario = lean_scenario_pb2.LeanScenario.FromString(raw_data)

            if self._include_map:
                map_data = lean_map_pb2.LeanMapContainer.FromString(raw_data)

                def _resolver(
                    key: MapKey | None = None,  # noqa: ARG001
                    _map_data: LeanMapContainer = map_data,
                ) -> MapGraph:
                    return WaymoMapGraphBuilder.from_proto(_map_data.map_features).build()

                resolver: MapResolver = _resolver
            else:
                resolver = no_map()

            yield _scenario_to_polars(scenario).lazy(), resolver

    @override
    def pipeline(self) -> Pipeline:
        resampling = self.loader_config.resampling
        no_resampling = resampling is None or resampling.no_resampling

        return (
            Pipeline()
            .compose(
                trajectory_pipeline(
                    self.loader_config,
                    derivative_rename=self.derivative_names(),
                    # When not using spline resampling, derivatives are computed
                    # separately below since vx/vy already exist in the raw data
                    # and only ax/ay need to be derived.
                    add_derivative=not no_resampling,
                    add_second_derivative=not no_resampling,
                )
            )
            .then(
                tr.derivative(
                    "vx",
                    "vy",
                    dt=self.loader_config.sample_time,
                    derivative_rename={1: ["ax", "ay"]},
                ),
                when=no_resampling,
            )
            # Shift autonomous vehicle id from -1 to 0
            .then(tr.with_columns(pl.col("id") + 1))
        )

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(input_len=10, output_len=80, sample_time=0.1).with_filtering(
            require_frames=[9]
        )


def _scenario_to_polars(scenario: lean_scenario_pb2.LeanScenario) -> pl.DataFrame:
    ego_track_index = scenario.sdc_track_index
    cat_map = _OBJECT_TYPE_TO_CATEGORY

    # Pre-fetch lists to avoid dictionary lookups in the inner loop
    l_frame = []
    l_tid = []
    l_x = []
    l_y = []
    l_vx = []
    l_vy = []
    l_yaw = []
    l_cat = []

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
            "x": pl.Float32,
            "y": pl.Float32,
            "vx": pl.Float32,
            "vy": pl.Float32,
            "yaw": pl.Float32,
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
