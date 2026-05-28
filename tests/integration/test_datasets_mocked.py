# pyright: standard
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dronalize.config import RuntimeOverride
from dronalize.io import StorageBackend
from dronalize.runtime import ExecutionRequest, resolve_request
from tests.integration.assertions import assert_plan_scene_outputs
from tests.support import demo_descriptor

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("jobs", [1, 4], ids=["jobs=1", "jobs=4"])
def test_datasets_mocked_registry_smoke(
    jobs: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    result = assert_plan_scene_outputs(plan, dataset_name="demo")

    assert result.checked_scenes > 0
