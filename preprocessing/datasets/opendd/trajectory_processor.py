from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from matplotlib.pyplot import table
from typing_extensions import override

from preprocessing.core import AgentCategory
from preprocessing.core.interface import DataProcessor, ProcessorConfig, Resampling

if TYPE_CHECKING:
    from collections.abc import Iterable


class OpenDDProcessor(DataProcessor[str, str]):
    def __init__(self, database_path: Path, config: ProcessorConfig | None = None) -> None:
        super().__init__(enforce_schema=True, processor_config=config)
        self._conn = sqlite3.connect(database_path)
        self._cursor = self._conn.cursor()

    @override
    def sources(self) -> Iterable[tuple[str, str]]:
        self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for row in self._cursor.fetchall():
            yield row[0], row[0]

    @override
    def load_raw(self, source: str) -> Iterable[pl.LazyFrame]:
        # Possible to include: UTM_ANGLE, V, ACC, ACC_LAT, ACC_TAN
        query = f"""
        SELECT
            ID as id,
            TIMESTAMP,
            UTM_X as x,
            UTM_Y as y,
            ACC_TAN,
            CLASS
        FROM {source}
        """  # noqa: S608
        df = (
            pl
            .read_database(query, self._conn)
            .lazy()
            .with_columns([
                (pl.col("TIMESTAMP").round(4).rank(method="dense") - 1)
                .cast(pl.Int64)
                .alias("frame"),
                pl
                .when(pl.col("CLASS").is_in(["Car", "Medium Vehicle"]))
                .then(AgentCategory.CAR)
                .when(pl.col("CLASS").is_in(["Heavy Vehicle", "Trailer"]))
                .then(AgentCategory.TRUCK)
                .when(pl.col("CLASS") == "Bus")
                .then(AgentCategory.BUS)
                .when(pl.col("CLASS") == "Motorcycle")
                .then(AgentCategory.MOTORCYCLE)
                .when(pl.col("CLASS") == "Pedestrian")
                .then(AgentCategory.PEDESTRIAN)
                .when(pl.col("CLASS") == "Bicycle")
                .then(AgentCategory.BICYCLE)
                .otherwise(AgentCategory.UNKNOWN)
                .alias("agent_category"),
            ])
            .drop("TIMESTAMP", "CLASS")
        )

        yield df

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame:
        print(df.collect())
        raise NotImplementedError

    @override
    def default_config(self) -> ProcessorConfig:
        ProcessorConfig(60, 150, 1 / 30).resampling_parameters(1, 3).window_parameters(75)


if __name__ == "__main__":
    path = Path("data/example_data/rdb1_4.sqlite")
    processor = OpenDDProcessor(path)
    for scene in processor.scenes_iter():
        print(scene.inner)
