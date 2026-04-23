"""Loader implementation for the INTERACTION dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory, DatasetSplit
from dronalize.core.functional import yaw_from_vel_expr
from dronalize.core.scene.schema import POSITIONS_VELOCITY_YAW
from dronalize.processing.loading.base import BaseSceneLoader
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.processing.maps.resolver import MapResolver, no_map, shared_map

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.scene import TrajectorySchema
    from dronalize.processing.loading.resources import DatasetResources
    from dronalize.processing.models import LoaderRequest


_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)


class InteractionLoader(BaseSceneLoader[Path]):
    """Loader for the INTERACTION dataset."""

    def __init__(
        self,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> None:
        super().__init__(data_root=data_root, request=request, resources=resources)

    @classmethod
    @override
    def unified_factory(
        cls,
        data_root: Path | str,
        request: LoaderRequest,
        resources: DatasetResources | None = None,
    ) -> InteractionLoader:
        return cls(data_root, request, resources)

    def _sources_from_dir(self, data_dir: Path) -> Iterable[Source[Path]]:
        if not data_dir.is_dir():
            return
        for file in sorted(data_dir.glob("*.csv")):
            yield Source(
                identifier=file.stem, data=file, map_key=file.stem.rstrip(self._strip(file.stem))
            )

    @staticmethod
    def _strip(file: str) -> str:
        """Strip the suffix from a file name to get the map key."""
        if "train" in file:
            return "_train"
        if "val" in file:
            return "_val"
        if "obs" in file:
            return "_obs"
        msg = f"Unexpected file name: {file}"
        raise ValueError(msg)

    @override
    def iter_sources_for(self, split: DatasetSplit) -> Iterable[Source[Path]]:
        if split is DatasetSplit.TRAIN:
            yield from self._sources_from_dir(self.root / "train")
        elif split is DatasetSplit.VAL:
            yield from self._sources_from_dir(self.root / "val")
        elif split is DatasetSplit.TEST:
            yield from self._sources_from_dir(self.root / "test_multi-agent")
            yield from self._sources_from_dir(self.root / "test_conditional-multi-agent")

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        data = (
            pl
            .scan_csv(source.data, schema=_SCHEMA)
            .drop("track_to_predict", "interesting_agent", "width", "length", "timestamp_ms")
            .rename({"frame_id": "frame", "track_id": "id", "psi_rad": "yaw_rad"})
            .with_columns(
                pl.col("case_id").cast(pl.UInt32),
                self._map_agent_category().alias("agent_category"),
                pl
                .when(pl.col("yaw_rad").is_null())
                .then(yaw_from_vel_expr())
                .otherwise(pl.col("yaw_rad"))
                .alias("yaw"),
            )
            .drop("agent_type")
        )

        for _, group in data.collect().group_by(["case_id"]):
            yield LoadedSourceData(frame=group.lazy().drop("case_id"))

    @override
    def count_sources_for(self, split: DatasetSplit) -> int | None:
        if split is DatasetSplit.TRAIN:
            return self._count_sources(self.root / "train")
        if split is DatasetSplit.VAL:
            return self._count_sources(self.root / "val")
        return self._count_sources(self.root / "test_multi-agent") + self._count_sources(
            self.root / "test_conditional-multi-agent"
        )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_VELOCITY_YAW

    @staticmethod
    def _map_agent_category() -> pl.Expr:
        return (
            pl
            .when(pl.col("agent_type") == "car")
            .then(AgentCategory.CAR.value)
            .when(pl.col("agent_type").is_in(["pedestrian/bicycle"]))
            .then(
                pl
                .when((pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt() < 4)
                .then(AgentCategory.PEDESTRIAN.value)
                .otherwise(AgentCategory.BICYCLE.value)
            )
            .otherwise(pl.col("agent_type"))
        )

    @staticmethod
    def _count_sources(data_dir: Path) -> int:
        if not data_dir.is_dir():
            return 0
        return sum(1 for _ in data_dir.rglob("*.csv"))

    @override
    def map_resolver(self) -> MapResolver:
        shared_maps = self.resources.shared_maps
        if not shared_maps or self.map_config is None:
            return no_map()
        return shared_map(shared_maps)


_SCHEMA = pl.Schema({
    "case_id": pl.Float64,
    "track_id": pl.UInt32,
    "frame_id": pl.UInt32,
    "timestamp_ms": pl.Float64,
    "agent_type": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "psi_rad": pl.Float64,
    "length": pl.Float64,
    "width": pl.Float64,
    "track_to_predict": pl.Float64,
    "interesting_agent": pl.Float64,
})
