# pyright: standard
from __future__ import annotations

import itertools
import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np
import pytest

from dronalize.config import RuntimeOverride
from dronalize.io import StorageBackend
from dronalize.io.encoding.common import encode_scene_record
from dronalize.runtime import ExecutionRequest, resolve_request
from dronalize.runtime.api import stream_plan
from tests.integration.assertions import (
    assert_basic_map_sanity,
    assert_basic_scene_sanity,
    assert_record_sanity,
)
from tests.support import demo_descriptor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("jobs", [1, 4], ids=["jobs=1", "jobs=4"])
def test_datasets_mocked_registry_smoke(
    jobs: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if jobs > 1:
        mp.set_start_method("spawn", force=True)

    monkeypatch.setattr("dronalize.runtime.api.get_dataset", lambda _name: demo_descriptor())

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

    n_scenes = 0
    for scene in itertools.islice(stream_plan(plan), 2):
        assert_basic_scene_sanity(scene)
        graph = scene.resolve_map()
        assert_basic_map_sanity(graph, expect_map=scene.has_map())
        record = encode_scene_record(
            scene, dtype=np.float32, trajectory_schema=plan.output.trajectory_schema
        )
        assert_record_sanity(record, scene)
        n_scenes += 1

    assert n_scenes > 0
