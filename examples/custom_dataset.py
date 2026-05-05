"""Minimal custom dataset integration for dronalize.

The raw dataset is one CSV file with the required `positions_only` trajectory
fields: `frame`, `id`, `x`, `y`, and `agent_category`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.config.models import DatasetConfig, ScenesConfig, WindowConfig
from dronalize.core import AgentCategory
from dronalize.core.scene import POSITIONS_ONLY, TrajectorySchema
from dronalize.datasets import DatasetSpec, register
from dronalize.processing.loading import BaseSceneLoader, LoadedSourceData, Source
from dronalize.runtime import ExecutionRequest, resolve_request

if TYPE_CHECKING:
    from collections.abc import Iterable


DATA: str = """frame,id,x,y,category
0,1,0.0,0.0,car
1,1,0.5,0.0,car
2,1,1.0,0.0,car
3,1,1.5,0.0,car
4,1,2.0,0.0,car
"""


class MiniCsvLoader(BaseSceneLoader[str]):
    @override
    def iter_sources(self) -> Iterable[Source[str]]:
        # Only one source for this simple example.
        yield Source(identifier="", data=DATA)

    @override
    def load_source(self, source: Source[str]) -> Iterable[LoadedSourceData]:
        # Should normalize the raw data to match the expected native schema,
        # which in this case includes mapping string categories to
        # `AgentCategory` enums, and renaming the `category` column to
        # `agent_category`.
        frame = pl.scan_csv(source.data, schema=_RAW_SCHEMA).with_columns(
            pl
            .col("category")
            .replace_strict({"car": AgentCategory.CAR, "pedestrian": AgentCategory.PEDESTRIAN})
            .alias("agent_category")
        )
        yield LoadedSourceData(frame)

    @override
    def count_sources(self) -> int | None:
        return 1

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return POSITIONS_ONLY


MINI_CSV_SPEC = DatasetSpec(
    name="mini-csv",
    loader_factory=MiniCsvLoader.unified_factory,
    default_config=DatasetConfig(
        scenes=ScenesConfig(
            history_frames=2, future_frames=1, sample_time=0.1, window=WindowConfig(step=1)
        )
    ),
    native_schema=MiniCsvLoader.native_trajectory_schema(),
)


def register_dronalize_datasets() -> DatasetSpec:
    """CLI hook used to register dataset."""
    return MINI_CSV_SPEC


_RAW_SCHEMA = pl.Schema({
    "frame": pl.Int64,
    "id": pl.Int64,
    "x": pl.Float64,
    "y": pl.Float64,
    "category": pl.Utf8,
})

if __name__ == "__main__":
    register(MINI_CSV_SPEC)
    request = ExecutionRequest(
        # Dataset string should be same as `MINI_CSV_SPEC.name` to resolve to
        # the correct dataset.
        dataset="mini-csv",
        # Input directory is unused in this simple example, since the loader
        # reads from hardcoded inline data. Otherwise, this should be the path
        # to the raw data files, available in property `self.root` of the loader.
        input_dir=Path(),
        output_dir=Path("./output"),
    )
    plan = resolve_request(request)
    # Inspect that the plan matches expectations
    print(f"dataset: {plan.dataset}")
    print(f"input: {plan.data_root}")
    print(f"output: {plan.output_dir}")
    print(f"schema: {plan.descriptor.native_schema.name}")
