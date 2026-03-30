from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.scene import CANONICAL_V1
from dronalize.processing.filters import Filter
from dronalize.processing.filters.agent import RequireFrames
from dronalize.processing.ingest.base import BaseSceneLoader, LoaderSplitCapabilities
from dronalize.processing.ingest.config import LoaderConfig
from dronalize.processing.ingest.loader import IngestedData, Source
from dronalize.processing.maps.config import MapConfig
from dronalize.processing.maps.resolver import MapResolver, no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.scene import SceneSchema
    from dronalize.processing.ingest.splits import SplitRequest


# NOTE: The dataset directory layout may later be aligned with the upstream
# one in repo: https://github.com/SOTIF-AVLab/SinD


class SindLoader(BaseSceneLoader[Path]):
    """Loader for the SIND dataset."""

    split_capabilities: ClassVar[LoaderSplitCapabilities] = LoaderSplitCapabilities(
        supports_source_split=True
    )

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
        split_request: SplitRequest | None = None,
    ) -> None:
        """Initialize the SindLoader.

        Parameters
        ----------
        data_root : Path or str
            Path to the directory containing the SIND dataset.
        loader_config : LoaderConfig, optional
            Loader configuration override.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            split_request=split_request,
        )
        self._data_root: Path = Path(data_root) / "data"

    @override
    def discover_sources(self) -> Iterable[Source[Path]]:
        if not self._data_root.is_dir():
            return
        for subdir in sorted(path for path in self._data_root.iterdir() if path.is_dir()):
            map_location = self._resolve_map_key(subdir.name)
            yield Source(identifier=subdir.name, data=subdir, map_key=str(map_location))

    @override
    def ingest(self, source: Source[Path]) -> Iterable[IngestedData]:
        subdir = source.data
        pedestrian_data_path = subdir / "Ped_smoothed_tracks.csv"
        vehicle_data_path = subdir / "Veh_smoothed_tracks.csv"
        vehicle_df = pl.scan_csv(vehicle_data_path, schema_overrides=_VEHICLE_SCHEMA).select(
            pl.col("track_id").alias("id"),
            pl.col("frame_id").alias("frame"),
            pl
            .col("agent_type")
            .replace_strict({
                "motorcycle": AgentCategory.MOTORCYCLE.value,
                "car": AgentCategory.CAR.value,
                "truck": AgentCategory.TRUCK.value,
                "tricycle": AgentCategory.TRICYCLE.value,
                "bus": AgentCategory.BUS.value,
                "bicycle": AgentCategory.BICYCLE.value,
            })
            .alias("agent_category"),
            pl.col("heading_rad").alias("yaw"),
            *("x", "y", "vx", "vy", "ax", "ay"),
        )

        pedestrian_df = pl.scan_csv(
            pedestrian_data_path, schema_overrides=_PEDESTRIAN_SCHEMA
        ).select(
            pl
            .col("track_id")
            .str.slice(1)
            .cast(pl.Int32)
            .add(vehicle_df.select(pl.col("id").max()).collect().item() + 1)
            .alias("id"),
            pl.col("frame_id").alias("frame"),
            pl
            .col("agent_type")
            .replace_strict({
                "pedestrian": AgentCategory.PEDESTRIAN.value,
                "animal": AgentCategory.ANIMAL.value,
            })
            .alias("agent_category"),
            pl.lit(None).alias("yaw"),
            *("x", "y", "vx", "vy", "ax", "ay"),
        )

        yield IngestedData(pl.concat([vehicle_df, pedestrian_df]))

    @override
    def num_sources(self) -> int | None:
        if not self._data_root.is_dir():
            return 0
        return sum(1 for path in self._data_root.iterdir() if path.is_dir())

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return CANONICAL_V1

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=20, output_len=50, sample_time=0.1)
            .with_window(25)
            .with_filter(Filter.define(agent_rules=[RequireFrames.define(frames=[19])]))
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.full_map()

    @override
    def map_resolver(self) -> MapResolver:
        if self._shared_memory_name is None:
            return no_map()
        return shared_map(self._shared_memory_name)

    @staticmethod
    def _resolve_map_key(path_name: str) -> str:
        if path_name.startswith("changchun"):
            return "changchun"
        if path_name.startswith("xian"):
            return "xian"
        if "NR" in path_name:
            return "nr_ll2"
        return "map_relink_law_save"


_VEHICLE_SCHEMA = pl.Schema({
    "track_id": pl.Int32,
    "frame_id": pl.Int32,
    "agent_type": pl.Utf8,
    "heading_rad": pl.Float64,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "ax": pl.Float64,
    "ay": pl.Float64,
})

_PEDESTRIAN_SCHEMA = pl.Schema({
    "track_id": pl.Utf8,
    "frame_id": pl.Int32,
    "agent_type": pl.Utf8,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "ax": pl.Float64,
    "ay": pl.Float64,
})


if __name__ == "__main__":
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env
    from dronalize.datasets.sind import DESCRIPTOR

    root = resolve_dataset_root_from_env("sind")
    _ = debug_descriptor(DESCRIPTOR, root)
