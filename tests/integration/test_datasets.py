from __future__ import annotations

import itertools
import multiprocessing as mp
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from dronalize.config import RuntimeOverride
from dronalize.datasets import list_datasets
from dronalize.io import StorageBackend
from dronalize.io.encoding.common import encode_scene_record
from dronalize.runtime import ExecutionRequest, resolve_request
from dronalize.runtime.api import stream_plan
from tests.integration.assertions import (
    assert_basic_map_sanity,
    assert_basic_scene_sanity,
    assert_record_sanity,
    save_scene_artifacts,
)

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
@pytest.mark.parametrize("jobs", [1, 2], ids=["jobs=1", "jobs=2"])
def test_datasets(
    case: DatasetCase, jobs: int, raw_data_root: Path, artifact_dir: Path, tmp_path: Path
) -> None:
    if not (raw_data_root / case.path_rel_root).exists():
        pytest.skip(f"Dataset root not found: {raw_data_root / case.path_rel_root}")
    if jobs > 1:
        case.max_scenes = 2
        case.scene_step = 10
        mp.set_start_method("spawn", force=True)

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
    n_scenes = 0
    for scene in itertools.islice(stream_plan(plan), case.scene_start, None, case.scene_step):
        assert_basic_scene_sanity(scene)
        graph = scene.resolve_map()
        assert_basic_map_sanity(graph, expect_map=scene.has_map())
        record = encode_scene_record(
            scene, dtype=np.float32, trajectory_schema=plan.output.trajectory_schema
        )
        assert_record_sanity(record, scene)
        if jobs == 1:
            save_scene_artifacts(
                scene=scene,
                graph=graph,
                out_dir=artifact_dir / case.dataset / f"scene_{scene.scene_number:03d}",
                dataset_name=case.dataset,
            )
        n_scenes += 1
        if n_scenes >= case.max_scenes:
            break
    assert 0 < n_scenes <= case.max_scenes
