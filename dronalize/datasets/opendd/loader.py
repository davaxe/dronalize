from __future__ import annotations

import functools
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config import LoaderConfig
from dronalize.config.map import MapConfig
from dronalize.datasets.common import utils
from dronalize.datasets.opendd.map.builder import OpenDDMapBuilder
from dronalize.loading import BaseSceneLoader
from dronalize.loading.loader import IngestOutput, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.maps.graph import MapGraph
    from dronalize.maps.resolver import MapKey, MapResolver
    from dronalize.scene import Scene


def _table_query(table_name: str) -> str:
    return f"""
    SELECT
        OBJID as id,
        TIMESTAMP,
        UTM_X as x,
        UTM_Y as y,
        CLASS
    FROM {table_name}
    """


class OpenDDLoader(BaseSceneLoader[tuple[Path, str]]):
    """Loader for OpenDD data split across multiple SQLite databases."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        *,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        """Initialize the multi-database OpenDD loader.

        Parameters
        ----------
        data_root : Path or str
            Path to a directory containing one subdirectory or database file per
            OpenDD segment.
        loader_config : , optional
            Loader configuration override. If None, the default configuration
            is used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` processes all sources.

        """
        super().__init__(
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            enforce_schema=True,
        )
        self._data_root: Path = self._normalize_data_root(data_root)

    def _db_paths(self) -> Iterable[Path]:
        for db_file in self._data_root.rglob("trajectories_*_v3.sqlite"):
            level = len(db_file.relative_to(self._data_root).parts)
            if level > 3:
                continue
            yield db_file

    @override
    def discover_sources(self) -> Iterable[Source[tuple[Path, str]]]:
        for db_path in self._db_paths():
            for table_name in _list_table_names(db_path):
                identifier = f"{db_path.stem}:{table_name}"
                yield Source(identifier=identifier, inner=(db_path, table_name))

    @override
    def num_sources(self) -> int | None:
        return sum(_count_tables(db_path) for db_path in self._db_paths())

    @override
    def ingest(self, source: Source[tuple[Path, str]]) -> Iterable[IngestOutput]:
        db_path, table_name = source.inner
        with sqlite3.connect(db_path) as connection:
            yield (
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
                .drop("CLASS", "TIMESTAMP"),
                str(db_path.parent),
            )

    @override
    def pipeline(self) -> Pipeline:
        return (
            Pipeline()
            .compose(
                trajectory_pipeline(self.loader_config, derivative_rename=self.derivative_names())
            )
            .then(tr.yaw_from_vel())
        )

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=60, output_len=150, sample_time=1 / 30)
            .with_resampling(1, 3)
            .with_window(75)
            .with_filtering(require_frames=[59])
        )

    @classmethod
    @override
    def default_map_config(cls) -> MapConfig:
        return MapConfig.default()

    @override
    def map_resolver(self) -> MapResolver:
        def _resolver(scene: Scene, key: MapKey = None) -> MapGraph | None:
            if key is None:
                return None

            return utils.extract_based_on_scene(
                self._get_map(key, self.map_config.min_distance, self.map_config.interp_distance),
                scene,
                self.map_config.extraction,
            )

        return _resolver

    @staticmethod
    @functools.lru_cache(maxsize=10)
    def _get_map(
        key: str,
        min_distance: float | None,
        interp_distance: float | None,
    ) -> MapGraph:
        database: str = Path(key).name
        print(database)
        map_path = Path(key) / f"map_{database}" / f"map_{database}.sqlite"
        return OpenDDMapBuilder.from_sqlite_file(map_path).build(min_distance, interp_distance)


def _list_table_names(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    return [row[0] for row in rows]


def _count_tables(db_path: Path) -> int:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table';"
        ).fetchone()
    return row[0] if row is not None else 0
