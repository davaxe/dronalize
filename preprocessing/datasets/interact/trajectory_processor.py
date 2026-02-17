from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import override

import polars as pl

from preprocessing.core.interface import (
    DataProcessor,
    ProcessorConfig,
    Resampling,
)


@dataclass
class _Source:
    csv_files: list[Path]


class InteractionProcessor(DataProcessor[str, _Source]):
    def __init__(self, data_dir: Path, config: ProcessorConfig | None = None) -> None:
        super().__init__(config)
        self._data_dir = data_dir

    @override
    def sources(self) -> Iterable[tuple[str, _Source]]:
        csv_files = list(self._data_dir.glob("*.csv"))
        yield str(self._data_dir), _Source(csv_files=csv_files)

    @override
    def load_raw(self, source: _Source) -> Iterable[pl.LazyFrame]:
        yield (
            pl
            .scan_csv(
                *[source.csv_files],
                include_file_paths="file_id",
                schema=_SCHEMA,
            )
            .drop("track_to_predict", "interesting_agent", "width", "length", "timestamp_ms")
            .rename({
                "agent_type": "agent_category",
                "psi_rad": "yaw",
                "frame_id": "frame",
                "track_id": "id",
            })
            .with_columns(
                pl.col("file_id").cast(pl.Categorical).to_physical(),
                pl.col("case_id").cast(pl.UInt32),
            )
        )

    @override
    def normalize(self, df: pl.LazyFrame) -> pl.LazyFrame: ...

    @override
    def default_config(self) -> ProcessorConfig: ...


_SCHEMA = pl.Schema({
    "case_id": pl.Float32,
    "track_id": pl.UInt32,
    "frame_id": pl.UInt32,
    "timestamp_ms": pl.Float32,
    "agent_type": pl.String,
    "x": pl.Float32,
    "y": pl.Float32,
    "vx": pl.Float32,
    "vy": pl.Float32,
    "psi_rad": pl.Float32,
    "length": pl.Float32,
    "width": pl.Float32,
    "track_to_predict": pl.Float32,
    "interesting_agent": pl.Float32,
})


if __name__ == "__main__":
    processor = InteractionProcessor(data_dir=Path("data/interact"))
    for source_name, source in processor.sources():
        raw = processor.load_raw(source)
        print(f"Source: {source_name}")
        for df in raw:
            print(df.collect())
