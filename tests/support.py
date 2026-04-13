from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pydantic import Field
from typing_extensions import override

from dronalize.config.models import DatasetConfig, ScenesConfig, WindowConfig
from dronalize.core.maps import MapGraph
from dronalize.core.scene import CANONICAL, Scene, TrajectorySchema
from dronalize.datasets import DatasetSpec
from dronalize.io.records import SceneRecord
from dronalize.processing.loading.base import BaseSceneLoader, DatasetOptionsModel
from dronalize.processing.loading.loader import LoadedSourceData, Source

if TYPE_CHECKING:
    from collections.abc import Iterable


class DemoOptions(DatasetOptionsModel):
    batch_size: int = Field(default=2, gt=0)
    use_cache: bool = False


class DemoLoader(BaseSceneLoader[Path, DemoOptions]):
    @classmethod
    @override
    def loader_options_model(cls) -> type[DemoOptions]:
        return DemoOptions

    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return CANONICAL

    @override
    def discover_sources(self) -> Iterable[Source[Path]]:
        yield Source(identifier="source-1", data=self.root / "source.parquet")

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        _ = source
        frame = pl.DataFrame({
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "vx": [1.0, 1.0, 1.0],
            "vy": [0.0, 0.0, 0.0],
            "ax": [0.0, 0.0, 0.0],
            "ay": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "agent_category": [1, 1, 1],
        })
        yield LoadedSourceData(frame.lazy())

    @override
    def num_sources(self) -> int | None:
        return 1


def demo_descriptor() -> DatasetSpec:
    return DatasetSpec(
        name="demo",
        loader_factory=DemoLoader.unified_factory,
        default_config=DatasetConfig(
            scenes=ScenesConfig(
                history_frames=2, future_frames=1, sample_time=1.0, window=WindowConfig(step=1)
            ),
            dataset={"batch_size": 2, "use_cache": False},
        ),
        native_schema=CANONICAL,
        dataset_options_model=DemoOptions,
        has_map=True,
    )


def make_scene(*, passed_agent_ids: frozenset[int] | None = None) -> Scene:
    graph = MapGraph(
        node_positions=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        edge_indices=np.array([[0], [1]], dtype=np.int32),
        node_types=np.array([1, 2], dtype=np.int32),
        edge_types=np.array([3], dtype=np.int32),
    )

    frame = pl.DataFrame({
        "frame": [0, 1, 2, 0, 1, 2],
        "id": [10, 10, 10, 20, 20, 20],
        "x": [0.0, 1.0, 2.0, 0.0, 0.0, 0.0],
        "y": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
        "vx": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "vy": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "ax": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "ay": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "yaw": [0.0, 0.0, 0.0, 1.57, 1.57, 1.57],
        "agent_category": [1, 1, 1, 2, 2, 2],
    })

    return Scene(
        frame=frame,
        scene_number=7,
        history_frames=2,
        future_frames=1,
        schema=CANONICAL,
        sample_time=1.0,
        map_key="demo-map",
        map_resolver=lambda _scene, graph=graph: graph,
        passed_agent_ids=passed_agent_ids,
    )


def assert_scene_record_equal(actual: SceneRecord, expected: SceneRecord) -> None:
    for field in fields(SceneRecord):
        left = getattr(actual, field.name)
        right = getattr(expected, field.name)
        if isinstance(left, np.ndarray):
            assert isinstance(right, np.ndarray)
            if np.issubdtype(left.dtype, np.floating):
                np.testing.assert_allclose(left, right)
            else:
                np.testing.assert_array_equal(left, right)
        else:
            assert left == right
