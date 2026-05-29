from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from dronalize.config import RuntimeOverride
from dronalize.datasets import list_datasets
from dronalize.io import StorageBackend
from dronalize.runtime import ExecutionRequest, resolve_request
from tests.integration.assertions import assert_plan_scene_outputs

if TYPE_CHECKING:
    from pathlib import Path


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
@pytest.mark.parametrize("jobs", [1, 2], ids=["jobs=1", "jobs=4"])
def test_datasets(
    case: DatasetCase, jobs: int, raw_data_root: Path, artifact_dir: Path, tmp_path: Path
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
