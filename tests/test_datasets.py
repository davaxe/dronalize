from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from prejectory.config import RuntimeOverride
from prejectory.datasets import DatasetDescriptor, get_dataset, list_datasets
from prejectory.datasets.registry import dataset_id_for_name, dataset_names_by_id
from prejectory.io import StorageBackend
from prejectory.processing.screening.agent import AgentRequireFrames
from prejectory.runtime import ExecutionRequest, resolve_request
from prejectory.runtime.types import build_loader_plan
from tests.support import demo_descriptor
from tests.support_integration import assert_plan_scene_outputs

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("name", list_datasets())
def test_builtin_datasets_resolve(name: str) -> None:
    descriptor = get_dataset(name)
    assert isinstance(descriptor, DatasetDescriptor)
    assert descriptor.name == name


def test_builtin_dataset_ids_are_unique() -> None:
    names = dataset_names_by_id()

    assert len(names) == len(set(names))
    for expected_id, name in enumerate(names):
        assert dataset_id_for_name(name) == expected_id


@pytest.mark.parametrize("name", list_datasets())
def test_builtin_default_benchmark_task_requires_history_endpoint(name: str) -> None:
    descriptor = get_dataset(name)
    assert descriptor.default_config.task is None
    assert descriptor.default_task == "benchmark"
    task = descriptor.tasks["benchmark"]
    config = descriptor.default_config.model_copy(update={"task": task})
    task = config.task
    assert task is not None
    loader = build_loader_plan(descriptor=descriptor, resolved_config=config, include_map=False)
    screening = loader.screening

    assert screening is not None
    assert "min_observations" in screening.cleanup
    assert "prediction_history_endpoint" in screening.agents
    rule = screening.agents["prediction_history_endpoint"]
    assert isinstance(rule, AgentRequireFrames)
    assert rule.frames == frozenset({task.prediction_origin - 1})
    assert rule.require is not None
    assert rule.require.absolute == 1
    assert rule.require.relative is None


@pytest.mark.parametrize("name", list_datasets())
def test_builtin_datasets_have_temporal_support(name: str) -> None:
    descriptor = get_dataset(name)
    temporal = descriptor.temporal_support

    assert temporal is not None
    assert temporal.source_frame_bounds.min_frames is not None
    assert temporal.source_frame_bounds.max_frames is not None
    assert temporal.source_frame_bounds.min_frames <= temporal.source_frame_bounds.max_frames
    assert temporal.windowing.max_window_frames == temporal.source_frame_bounds.max_frames
    assert temporal.windowing.default_policy in temporal.windowing.supported_policies


@dataclass(slots=True)
class DatasetCase:
    dataset: str
    path_rel_root: str
    max_scenes: int = 5
    scene_start: int = 0
    scene_step: int = 100


ALL_CASES_DEFAULT: dict[str, DatasetCase] = {
    name: DatasetCase(name, path_rel_root=name) for name in list_datasets()
}


@pytest.mark.slow
@pytest.mark.dataset
@pytest.mark.parametrize("case", ALL_CASES_DEFAULT.values(), ids=ALL_CASES_DEFAULT.keys())
@pytest.mark.parametrize("jobs", [1, 2], ids=["jobs=1", "jobs=2"])
def test_dataset_raw_data_processing(
    case: DatasetCase,
    jobs: int,
    raw_data_root: Path,
    artifact_dir: Path,
    tmp_path: Path,
) -> None:
    if not (raw_data_root / case.path_rel_root).exists():
        pytest.skip(f"Dataset root not found: {raw_data_root / case.path_rel_root}")
    if jobs > 1:
        case.max_scenes = 2
        case.scene_step = 10

    request = ExecutionRequest(
        dataset=case.dataset,
        input_dir=raw_data_root / case.dataset,
        output_dir=tmp_path,
        storage_backend=StorageBackend.NULL,
        overrides=RuntimeOverride.from_inputs(jobs=jobs),
        limit=case.max_scenes * case.scene_step + case.scene_start,
        include_map=jobs == 1,
    )
    plan = resolve_request(request)
    result = assert_plan_scene_outputs(
        plan,
        artifact_dir=artifact_dir / case.dataset if jobs == 1 else None,
        dataset_name=case.dataset,
        scene_start=case.scene_start,
        scene_step=case.scene_step,
    )
    assert 0 < result.checked_scenes <= case.max_scenes


@pytest.mark.parametrize("jobs", [1, 4], ids=["jobs=1", "jobs=4"])
def test_datasets_mocked_registry_smoke(
    jobs: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("prejectory.runtime.api.get_dataset", lambda _: demo_descriptor())  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    request = ExecutionRequest(
        dataset="demo",
        input_dir=input_dir,
        output_dir=output_dir,
        storage_backend=StorageBackend.NULL,
        overrides=RuntimeOverride.from_inputs(jobs=jobs),
        include_map=False,
        limit=2,
    )
    plan = resolve_request(request)
    result = assert_plan_scene_outputs(plan, dataset_name="demo")
    assert result.checked_scenes > 0
