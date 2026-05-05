from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pydantic import Field
from typing_extensions import NotRequired, TypedDict, override

from dronalize.config.models import DatasetConfig, ScenesConfig, WindowConfig
from dronalize.config.models.scenes import LaneChangeConfig, ResampleConfig
from dronalize.config.models.screening import MinSamplesSpec, ScreeningConfig
from dronalize.core.categories import AgentCategory, AgentCategoryLike
from dronalize.core.maps import MapGraph
from dronalize.core.scene import CANONICAL, Scene, TrajectorySchema
from dronalize.datasets import DatasetSpec
from dronalize.io.records import SceneRecord
from dronalize.processing.loading.base import BaseSceneLoader
from dronalize.processing.loading.loader import LoadedSourceData, Source
from dronalize.processing.loading.options import DatasetOptionsModel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


class AgentData(TypedDict):
    id: int
    x: Sequence[float]
    y: Sequence[float]
    category: AgentCategoryLike
    frame: NotRequired[Sequence[int]]
    vx: NotRequired[Sequence[float]]
    vy: NotRequired[Sequence[float]]
    ax: NotRequired[Sequence[float]]
    ay: NotRequired[Sequence[float]]
    yaw: NotRequired[Sequence[float]]
    lane_id: NotRequired[Sequence[int]]


class DemoOptions(DatasetOptionsModel):
    batch_size: int = Field(default=2, gt=0)
    use_cache: bool = False


class DemoLoader(BaseSceneLoader[Path, DemoOptions]):
    @classmethod
    @override
    def native_trajectory_schema(cls) -> TrajectorySchema:
        return CANONICAL

    @override
    def iter_sources(self) -> Iterable[Source[Path]]:
        yield Source(identifier="source-1", data=self.root / "source.parquet")

    @override
    def load_source(self, source: Source[Path]) -> Iterable[LoadedSourceData]:
        _ = source
        frame = pl.DataFrame(
            {
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
            },
            schema_overrides={"frame": pl.Int32(), "id": pl.Int32(), "agent_category": pl.Int32()},
        )
        yield LoadedSourceData(frame.lazy())

    @override
    def count_sources(self) -> int | None:
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


def inherited_optional_blocks_descriptor() -> DatasetConfig:
    return DatasetConfig(
        scenes=ScenesConfig(
            history_frames=2,
            future_frames=1,
            sample_time=1.0,
            window=WindowConfig(step=2),
            resample=ResampleConfig(up=2, down=1, method="cubic"),
            lane_change=LaneChangeConfig(persist=3),
        ),
        screening=ScreeningConfig(agent={"min_obs": MinSamplesSpec(minimum=2)}),
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

    return Scene.create(
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


def make_scene_df(*agent_data: AgentData) -> pl.DataFrame:
    optional_fields = {"frame", "vx", "vy", "ax", "ay", "yaw"}
    per_agent_rows: list[pl.DataFrame] = []
    present_optional_fields = {
        field for agent in agent_data for field in optional_fields if field in agent
    }
    for agent in agent_data:
        n = len(agent["x"])
        data = {
            "frame": _get_frames(agent),
            "id": [agent["id"]] * n,
            "x": agent["x"],
            "y": agent["y"],
            "agent_category": [AgentCategory.from_value(agent["category"]).value] * n,
        }
        for field in present_optional_fields:
            data[field] = agent.get(field, [None] * n)

        per_agent_rows.append(pl.DataFrame(data))

    return pl.concat(per_agent_rows, how="vertical") if per_agent_rows else pl.DataFrame()


def _get_frames(agent: AgentData) -> list[int]:
    samples = len(agent["x"])
    if "frame" in agent:
        return list(agent["frame"])
    return list(range(samples))


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


class DataFramePresets(TypedDict):
    single_agent: Callable[[], pl.DataFrame]
    single_agent_windowed: Callable[[], pl.DataFrame]
    single_agent_time_split: Callable[[], pl.DataFrame]
    two_agents: Callable[[], pl.DataFrame]
    lane_change_sequences: Callable[[], pl.DataFrame]


def scene_df_presets() -> DataFramePresets:
    return {
        "single_agent": lambda: make_scene_df(
            AgentData(id=1, x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0], category="car", frame=[0, 1, 2])
        ),
        "single_agent_windowed": lambda: make_scene_df(
            AgentData(
                id=1,
                x=[0.0, 1.0, 2.0, 3.0],
                y=[0.0, 0.0, 0.0, 0.0],
                category="car",
                frame=[0, 1, 2, 3],
            )
        ),
        "single_agent_time_split": lambda: make_scene_df(
            AgentData(
                id=1,
                x=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                y=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                category="car",
                frame=[0, 1, 2, 3, 4, 5, 6, 7],
            )
        ),
        "two_agents": lambda: make_scene_df(
            AgentData(id=1, x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0], category="car", frame=[0, 1, 2]),
            AgentData(
                id=2, x=[5.0, 5.5, 6.0], y=[5.0, 5.5, 6.0], category="pedestrian", frame=[0, 1, 2]
            ),
        ),
        "lane_change_sequences": lambda: pl.DataFrame({
            "sequence": [
                "changing",
                "changing",
                "changing",
                "changing",
                "changing",
                "steady",
                "steady",
                "steady",
                "steady",
                "steady",
            ],
            "frame": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "agent_category": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "lane_id": [1, 1, 2, 2, 2, 1, 1, 1, 1, 1],
        }),
    }
