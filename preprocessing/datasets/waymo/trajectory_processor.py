from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path
from tkinter import NO
from typing import TYPE_CHECKING, Literal, override

import polars as pl

# Assuming these exist in your project
from preprocessing.core import AgentCategory
from preprocessing.core.interface import DataProcessor, ProcessorConfig, Scene
from preprocessing.datasets.waymo.map.graph_builder import WaymoMapGraphBuilder
from preprocessing.datasets.waymo.protos import (
    lean_map_pb2,
    lean_scenario_pb2,
    scenario_pb2,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from preprocessing.core.map_graph import MapGraph

FilterStr = Literal["*.tfrecord*", "*training.tfrecord*"]  # abbreviated for brevity


class WaymoProcessor(DataProcessor[str, Path]):
    def __init__(
        self,
        data_dir: Path | str,
        filter_str: FilterStr,
        processor_config: ProcessorConfig | None = None,
        *,
        include_map: bool = True,
        interp_distance: float | None = None,
        min_distance: float | None = 1.5,
    ):
        super().__init__(
            processor_config=processor_config or self._default_config(),
            enforce_schema=False,
        )
        self._data_dir: Path = (
            Path(data_dir) if isinstance(data_dir, str) else data_dir
        )
        self._filter_str: FilterStr = filter_str
        self._include_map: bool = include_map
        self._interp_distance: float | None = interp_distance
        self._min_distance: float | None = min_distance
        self._current_map: MapGraph | None = None

    @override
    def sources(self) -> Iterable[tuple[str, Path]]:
        # Sorting ensures deterministic processing order
        for tfrecord_path in sorted(self._data_dir.glob(self._filter_str)):
            yield (tfrecord_path.stem, tfrecord_path)

    @override
    def load_raw(self, source: Path) -> Iterable[pl.LazyFrame]:
        for raw_data in _read_tfrecord(source):
            scenario = lean_scenario_pb2.LeanScenario.FromString(raw_data)

            # Map processing remains per-scenario (if needed)
            if self._include_map:
                map_data = lean_map_pb2.LeanMapContainer.FromString(raw_data)
                # Note: This overwrites _current_map repeatedly;
                # acceptable if modify_scene uses the map immediately after this yield.
                self._current_map = WaymoMapGraphBuilder.from_proto(
                    map_data.map_features
                ).build(
                    min_distance=self._min_distance,
                    interp_distance=self._interp_distance,
                )

            yield _scenario_to_polars(scenario).lazy()

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # TODO: 1. Filter
        #       2. Resample
        #       3. Add acceleration
        return df

    def modify_scene(self, scene: Scene) -> Scene:
        if self._include_map:
            return replace(scene, map=self._current_map)
        return scene

    @staticmethod
    def _default_config() -> ProcessorConfig:
        return ProcessorConfig(10, 80, 0.1)


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
            "category": l_cat,
        },
        schema={
            "frame": pl.Int32,
            "id": pl.Int32,
            "x": pl.Float32,
            "y": pl.Float32,
            "vx": pl.Float32,
            "vy": pl.Float32,
            "yaw": pl.Float32,
            "category": pl.Int32,
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


_OBJECT_TYPE_TO_CATEGORY: dict[int, AgentCategory] = {
    scenario_pb2.Track.TYPE_UNSET: AgentCategory.UNKNOWN,
    scenario_pb2.Track.TYPE_VEHICLE: AgentCategory.CAR,
    scenario_pb2.Track.TYPE_PEDESTRIAN: AgentCategory.PEDESTRIAN,
    scenario_pb2.Track.TYPE_CYCLIST: AgentCategory.BICYCLE,
    scenario_pb2.Track.TYPE_OTHER: AgentCategory.UNKNOWN,
}


if __name__ == "__main__":
    import time

    # Recommendation: Use ProcessPoolExecutor here for actual parallel processing
    # because Protobuf parsing + Python loops are CPU bound and single-threaded.
    directory = Path(
        "/home/west/Developer/behavior-prediction/datasets/waymo/training"
    )

    processor = WaymoProcessor(directory, "*training.tfrecord*")

    start_time = time.perf_counter()
    print("Starting processing...")
    count = 0
    # This loop will now yield one large  per file instead of per scene
    for scene in processor.scenes_iter():
        if count % 10 == 0:
            print(
                f"Processed {count} scenes in {time.perf_counter() - start_time:.2f}s"
            )
        count += 1

    print(
        f"Finished processing {count} scenes in {time.perf_counter() - start_time:.2f}s"
    )
