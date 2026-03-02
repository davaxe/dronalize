from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.common.trajectory.basic import yaw_from_vel
from dronalize.common.trajectory.process import prepare_agent_trajectories
from dronalize.core import AgentCategory, BaseSceneLoader, LoaderConfig
from dronalize.core.datatypes import map_context as mc
from dronalize.core.protocols.loader import Source

if TYPE_CHECKING:
    from collections.abc import Iterable


class OpenDDLoader(BaseSceneLoader[str, str]):
    """Processor for OpenDD dataset stored in SQLite format."""

    def __init__(self, database_path: Path, config: LoaderConfig | None = None) -> None:
        """Initialize the OpenDD processor.

        Parameters
        ----------
        database_path : Path
            Path to the OpenDD SQLite database file.
        config : LoaderConfig, optional
            Processor configuration override. If None, the default configuration
            will be used.

        """
        super().__init__(enforce_schema=True, loader_config=config)
        self._conn = sqlite3.connect(database_path)
        self._cursor = self._conn.cursor()

    @override
    def sources(self) -> Iterable[Source[str, str]]:
        self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for row in self._cursor.fetchall():
            yield Source(identifier=row[0], inner=row[0])

    @override
    def num_sources(self) -> int | None:
        self._cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
        return self._cursor.fetchone()[0]

    @override
    def load_raw(self, source: Source[str, str]) -> Iterable[tuple[pl.LazyFrame, mc.MapContext]]:
        table_name = source.inner
        # Possible to include: UTM_ANGLE, V, ACC, ACC_LAT, ACC_TAN
        query = f"""
        SELECT
            OBJID as id,
            TIMESTAMP,
            UTM_X as x,
            UTM_Y as y,
            CLASS
        FROM {table_name}
        """  # noqa: S608
        scenes = (
            pl
            .read_database(query, self._conn)
            .lazy()
            .with_columns([
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
            ])
            .drop("CLASS", "TIMESTAMP")
        )
        for df in prepare_agent_trajectories(
            scenes,
            config=self.loader_config,
            add_derivative=True,
            add_second_derivative=True,
            derivative_rename=self.derivative_names(),
        ):
            yield df, mc.Implicit()

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return yaw_from_vel(df, yaw_col="yaw")

    @override
    def default_config(self) -> LoaderConfig:
        return LoaderConfig(60, 150, 1 / 30).resampling_parameters(1, 3).window_parameters(75)


if __name__ == "__main__":
    path = Path("data/rdb2/trajectories_rdb2_v3.sqlite")
    processor = OpenDDLoader(path)
    count = 0
    for _scene in processor.scenes():
        count += 1
    print(f"Processed {count} scenes.")
