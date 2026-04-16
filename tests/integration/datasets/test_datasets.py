from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("altair")

from dronalize.config.runtime import RuntimeOverride
from dronalize.io import StorageBackend
from dronalize.io.encoding.common import encode_scene_record
from dronalize.runtime import ExecutionRequest, resolve_request
from dronalize.runtime.api import stream_plan
from tests.integration.datasets.assertions import (
    assert_basic_map_sanity,
    assert_basic_scene_sanity,
    assert_record_sanity,
    save_scene_artifacts,
)
from tests.integration.datasets.catalog import ALL_CASES_DEFAULT, DatasetCase

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.slow
@pytest.mark.dataset
@pytest.mark.parametrize("case", ALL_CASES_DEFAULT.values(), ids=ALL_CASES_DEFAULT.keys())
def test_datasets(
    case: DatasetCase, raw_data_root: Path, artifact_dir: Path, tmp_path: Path
) -> None:
    if not (raw_data_root / case.path_rel_root).exists():
        pytest.skip(f"Dataset root not found: {raw_data_root / case.path_rel_root}")

    request = ExecutionRequest(
        dataset=case.dataset,
        input_dir=raw_data_root / case.dataset,
        output_dir=tmp_path,
        storage_backend=StorageBackend.NULL,
        overrides=RuntimeOverride.from_inputs(jobs=1),
    )
    plan = resolve_request(request)
    n_scenes = 0
    for scene in itertools.islice(stream_plan(plan), case.scene_start, None, case.scene_step):
        assert_basic_scene_sanity(scene)
        graph = scene.resolve_map()
        assert_basic_map_sanity(graph, expect_map=scene.has_map())
        record = encode_scene_record(
            scene,
            dtype=np.float32,
            trajectory_schema=plan.output.trajectory_schema,
        )
        assert_record_sanity(record, scene)
        save_scene_artifacts(
            scene=scene,
            graph=graph,
            out_dir=artifact_dir / case.dataset / f"scene_{scene.scene_number:03d}",
            case=case,
        )
        n_scenes += 1
        if n_scenes >= case.max_scenes:
            break
    assert 0 < n_scenes <= case.max_scenes
