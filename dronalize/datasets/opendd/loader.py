from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

import dronalize.pipeline.transforms as tr
from dronalize.categories import AgentCategory, DatasetSplit
from dronalize.config import LoaderConfig
from dronalize.loading import BaseSceneLoader
from dronalize.loading.loader import IngestOutput, Source
from dronalize.pipeline.factories import trajectory_pipeline
from dronalize.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.config.map import MapConfig


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


class OpenDDLoader(BaseSceneLoader[str]):
    """Loader for OpenDD data stored in a single SQLite database."""

    def __init__(
        self,
        data_root: Path | str,
        loader_config: LoaderConfig | None = None,
        map_config: MapConfig | None = None,
        splits: Iterable[DatasetSplit] | DatasetSplit | None = None,
    ) -> None:
        """Initialize the OpenDD loader.

        Parameters
        ----------
        data_root : Path or str
            Path to the SQLite database file.
        loader_config : , optional
            Loader configuration override. If None, the default configuration
            is used.
        splits : Iterable[DatasetSplit] | DatasetSplit | None, optional
            Dataset split selection. This dataset does not define predefined
            splits, so `None` or `DatasetSplit.ALL` process all sources.

        """
        super().__init__(loader_config=loader_config, map_config=map_config, splits=splits)
        self._db_path: Path = self._normalize_data_root(data_root)
        self._conn: sqlite3.Connection | None = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
        return self._conn

    def __del__(self) -> None:
        """Best-effort cleanup for the lazily opened SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __getstate__(self) -> dict[str, object]:
        """Drop non-picklable connection state when the loader is serialized."""
        state = self.__dict__.copy()
        state["_conn"] = None
        return state

    def _table_names(self) -> list[str]:
        rows = (
            self
            ._connection()
            .execute("SELECT name FROM sqlite_master WHERE type='table';")
            .fetchall()
        )
        return [row[0] for row in rows]

    @override
    def all_sources(self) -> Iterable[Source[str]]:
        for table_name in self._table_names():
            yield Source(identifier=table_name, inner=table_name)

    @override
    def num_sources(self) -> int | None:
        row = (
            self
            ._connection()
            .execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
            .fetchone()
        )
        return row[0] if row is not None else 0

    @override
    def ingest(self, source: Source[str]) -> Iterable[IngestOutput]:
        yield (
            pl
            .read_database(_table_query(source.inner), self._connection(), iter_batches=False)
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
            None,
        )

    @override
    def pipeline(self) -> Pipeline:
        return _opendd_pipeline(self.loader_config, self.derivative_names())

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return (
            LoaderConfig(input_len=60, output_len=150, sample_time=1 / 30)
            .with_resampling(1, 3)
            .with_window(75)
            .with_filtering(require_frames=[59])
        )


class MultiOpenDDLoader(BaseSceneLoader[tuple[Path, str]]):
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
            splits, so `None` or `DatasetSplit.ALL` process all sources.

        """
        super().__init__(
            loader_config=loader_config,
            map_config=map_config,
            splits=splits,
            enforce_schema=True,
        )
        self._data_root: Path = self._normalize_data_root(data_root)

    def _db_paths(self) -> Iterable[Path]:
        if self._data_root.is_file():
            yield self._data_root
            return

        if not self._data_root.is_dir():
            return

        for child in sorted(self._data_root.iterdir()):
            if child.is_file() and child.suffix in {".db", ".sqlite", ".sqlite3"}:
                yield child
                continue

            if not child.is_dir():
                continue

            db_file = _find_sqlite_file(child)
            if db_file is not None:
                yield db_file

    @override
    def all_sources(self) -> Iterable[Source[tuple[Path, str]]]:
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
                None,
            )

    @override
    def pipeline(self) -> Pipeline:
        return _opendd_pipeline(self.loader_config, self.derivative_names())

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return OpenDDLoader.default_config()


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


def _find_sqlite_file(directory: Path) -> Path | None:
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in {".db", ".sqlite", ".sqlite3"}
    )
    if not candidates:
        return None
    return candidates[0]


def _opendd_pipeline(
    loader_config: LoaderConfig,
    derivative_names: dict[int, list[str]],
) -> Pipeline:
    return (
        Pipeline()
        .compose(trajectory_pipeline(loader_config, derivative_rename=derivative_names))
        .then(tr.yaw_from_vel())
    )
