"""Loader implementation for the OpenDD dataset."""

from __future__ import annotations

import functools
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.categories import AgentCategory
from dronalize.core.scene import POSITIONS_ONLY
from dronalize.datasets.opendd.maps import OpenDDMapBuilder
from dronalize.datasets.shared import utils
from dronalize.processing.loading.base import SceneLoader
from dronalize.processing.loading.models import DatasetSource, LoadedSourceFrame

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import Scene, TrajectorySchema
    from dronalize.processing.maps import MapResolver


def _table_query(table_name: str) -> str:
    return f"""
    SELECT
        OBJID as id,
        TIMESTAMP,
        UTM_X as x,
        UTM_Y as y,
        CLASS
    FROM {table_name}
    """  # noqa: S608


class OpenDDLoader(SceneLoader[tuple[Path, str]]):
    """Loader for OpenDD data split across multiple SQLite databases."""

    def _db_paths(self) -> Iterable[Path]:
        for db_file in self.root.rglob("trajectories_*_v3.sqlite"):
            level = len(db_file.relative_to(self.root).parts)
            if level <= 3:
                yield db_file

    @override
    def iter_sources(self) -> Iterable[DatasetSource[tuple[Path, str]]]:
        for db_path in self._db_paths():
            for table_name in _list_table_names(db_path):
                identifier = f"{db_path.stem}:{table_name}"
                yield DatasetSource(
                    identifier=identifier,
                    payload=(db_path, table_name),
                    map_key=str(db_path.parent),
                )

    @override
    def count_sources(self) -> int | None:
        return sum(_count_tables(db_path) for db_path in self._db_paths())

    @override
    def load_source(self, source: DatasetSource[tuple[Path, str]]) -> Iterable[LoadedSourceFrame]:
        db_path, table_name = source.payload
        with sqlite3.connect(db_path) as connection:
            yield LoadedSourceFrame(
                pl
                .read_database(_table_query(table_name), connection)
                .lazy()
                .with_columns(
                    ((pl.col("TIMESTAMP") * 1000).round(4).rank(method="dense") - 1)
                    .cast(pl.Int64)
                    .alias("frame"),
                    pl
                    .col("CLASS")
                    .replace_strict(
                        {
                            "Car": AgentCategory.CAR,
                            "Medium Vehicle": AgentCategory.CAR,
                            "Heavy Vehicle": AgentCategory.TRUCK,
                            "Trailer": AgentCategory.TRUCK,
                            "Bus": AgentCategory.BUS,
                            "Motorcycle": AgentCategory.MOTORCYCLE,
                            "Pedestrian": AgentCategory.PEDESTRIAN,
                            "Bicycle": AgentCategory.BICYCLE,
                        },
                        default=AgentCategory.UNKNOWN,
                    )
                    .alias("agent_category"),
                )
                .drop("CLASS", "TIMESTAMP")
            )

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY

    @override
    def map_resolver(self) -> MapResolver:
        def _resolver(scene: Scene) -> MapGraph | None:
            if scene.map_key is None or self.map_config is None:
                return None
            return utils.extract_configured_map(
                self._get_map(
                    scene.map_key,
                    self.map_config.min_distance,
                    self.map_config.interpolation_distance,
                ),
                scene,
                self.map_config,
            )

        return _resolver

    @staticmethod
    @functools.lru_cache(maxsize=10)
    def _get_map(
        key: str, min_distance: float | None, interpolation_distance: float | None
    ) -> MapGraph:
        database = Path(key).name
        map_path = Path(key) / f"map_{database}" / f"map_{database}.sqlite"
        return OpenDDMapBuilder.from_sqlite_file(map_path).build(
            min_distance, interpolation_distance
        )


def _list_table_names(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
        return [name for (name,) in cursor]


def _count_tables(db_path: Path) -> int:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table';"
        ).fetchone()
    return row[0] if row is not None else 0
