from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from typing_extensions import override

# Assuming these exist in your project
from preprocessing.common.trajectory_utils.derivative import derivative
from preprocessing.common.trajectory_utils.filter import filter_scene_expr
from preprocessing.common.trajectory_utils.resample import resample_tracks
from preprocessing.core import AgentCategory
from preprocessing.core import map_context as mc
from preprocessing.core.interface import BaseSceneLoader, LoaderConfig, Resampling
from preprocessing.datasets.waymo.map.graph_builder import WaymoMapGraphBuilder
from preprocessing.datasets.waymo.protos import lean_map_pb2, lean_scenario_pb2, scenario_pb2

if TYPE_CHECKING:
    from collections.abc import Iterable

FilterStr = Literal[
    "*training.tfrecord*",
    "*training_20s.tfrecord*",
    "*validation.tfrecord*",
    "validation_interactive.tfrecord*",
    "*testing.tfrecord*",
    "*testing_interactive.tfrecord*",
    "*.tfrecord*",
]


class WaymoLoader(BaseSceneLoader[str, Path]):
    """Processor for Waymo Open Dataset scenarios stored in TFRecord format."""

    def __init__(
        self,
        data_dir: Path | str,
        filter_str: FilterStr,
        loader_config: LoaderConfig | None = None,
        *,
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

        Args:
            data_dir: directory containing the TFRecord files.
            filter_str: string pattern to filter TFRecord files
                (e.g., "*validation.tfrecord*").
            loader_config: Configuration if not default.
            include_map: Whether to include map data in the scene. Defaults to True.
            interp_distance: Distance threshold for interpolating map points.
                Defaults to None (no interpolation).
            min_distance: Minimum distance between map points after processing.
                Defaults to 1.5 meters.

        """
        super().__init__(loader_config=loader_config, enforce_schema=True)
        self._data_dir: Path = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self._filter_str: FilterStr = filter_str
        self._include_map: bool = include_map
        self._interp_distance: float | None = interp_distance
        self._min_distance: float | None = min_distance

    @override
    def sources(self) -> Iterable[tuple[str, Path]]:
        # Sorting ensures deterministic processing order
        for tfrecord_path in sorted(self._data_dir.glob(self._filter_str)):
            yield (tfrecord_path.stem, tfrecord_path)

    @override
    def num_sources(self) -> int | None:
        return sum(1 for _ in self._data_dir.glob(self._filter_str))

    @override
    def load_raw(self, source: Path) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
        for i, raw_data in enumerate(_read_tfrecord(source)):
            scenario = lean_scenario_pb2.LeanScenario.FromString(raw_data)

            # Map processing remains per-scenario (if needed)
            map_context: mc.MapContext = mc.Explicit(str(source), record_index=i)
            if self._include_map:
                map_data = lean_map_pb2.LeanMapContainer.FromString(raw_data)
                current_map = WaymoMapGraphBuilder.from_proto(map_data.map_features).build(
                    min_distance=self._min_distance,
                    interp_distance=self._interp_distance,
                )
                map_context = mc.Loaded(current_map)

            yield _scenario_to_polars(scenario).lazy(), map_context

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        resampling = self.loader_config.resampling or Resampling(1, 1)
        df_filtered = df.filter(
            filter_scene_expr(
                self.loader_config,
                category_column="agent_category",
            )
        )
        df_resampled = resample_tracks(
            df_filtered,
            resampling.up,
            resampling.down,
            group_by=["id"],
            add_derivative=resampling.method == "spline",
            add_second_derivative=resampling.method == "spline",
            method=resampling.method,
            dt=self.loader_config.sample_time,
            derivative_rename=self.derivative_names(),
            forward_fill=["agent_category"],
        )

        if resampling.method != "spline":
            df_resampled = derivative(
                df_resampled,
                "vx",
                "vy",
                dt=self.loader_config.sample_time,
                derivative_rename={1: ["ax", "ay"]},
            )

        # Change autonomous vehicle to id 0 (from id -1)
        return df_resampled.with_columns(pl.col("id") + 1)

    @override
    def default_config(self) -> LoaderConfig:
        return LoaderConfig(10, 80, 0.1)


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
    directory = Path("/home/west/Developer/behavior-prediction/datasets/waymo/validation")

    processor = WaymoLoader(directory, "*validation.tfrecord*", include_map=True)
    start_time = time.perf_counter()
    print("Starting processing...")
    count = 0
    # This loop will now yield one large  per file instead of per scene
    for _scene in processor.scenes():
        if count % 500 == 0:
            print(f"Processed {count} scenes in {time.perf_counter() - start_time:.2f}s")
        count += 1
        print(_scene.map_context)
        break

    print(f"Finished processing {count} scenes in {time.perf_counter() - start_time:.2f}s")
