from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from dronalize.config.runtime import RuntimeOverride
from dronalize.io import SceneRecord, StorageBackend
from dronalize.io.encoding.common import encode_scene_record
from dronalize.runtime import ProcessRequest, resolve_job
from dronalize.runtime.api import run_job_yield
from tests.integration.datasets.catalog import ALL_CASES_DEFAULT, DatasetCase

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import Scene


def assert_basic_scene_sanity(scene: Scene, case: DatasetCase) -> None:
    assert scene.frame.height > 0
    assert scene.history_frames >= 0
    assert scene.future_frames >= 0


def assert_basic_map_sanity(graph: MapGraph | None, *, expect_map: bool) -> None:
    if expect_map:
        assert graph is not None
        assert graph.num_edges > 0
        assert graph.num_nodes > 0
    else:
        assert graph is None


def assert_record_sanity(record: SceneRecord, scene: Scene, case: DatasetCase) -> None: ...


def save_scene_artifacts(
    scene: Scene,
    graph: MapGraph | None,
    out_dir: Path,
    case: DatasetCase,
) -> None:
    """Save scene artifacts like trajectories and maps for debugging."""
    out_dir.mkdir(parents=True, exist_ok=True)


@pytest.mark.slow
@pytest.mark.dataset
@pytest.mark.parametrize("case", ALL_CASES_DEFAULT.values(), ids=ALL_CASES_DEFAULT.keys())
def test_datasets(
    case: DatasetCase, raw_data_root: Path, artifact_dir: Path, tmp_path: Path
) -> None:
    request = ProcessRequest(
        dataset=case.dataset,
        input_dir=raw_data_root / case.dataset,
        output_dir=tmp_path,
        storage_backend=StorageBackend.NULL,
        overrides=RuntimeOverride.from_inputs(jobs=1),
    )
    job = resolve_job(request)
    n_scenes = 0
    for scene in itertools.islice(run_job_yield(job), case.scene_start, None, case.scene_step):
        assert_basic_scene_sanity(scene, case)
        graph = scene.resolve_map()
        assert_basic_map_sanity(graph, expect_map=scene.has_map())
        record = encode_scene_record(
            scene,
            dtype=np.float32,
            trajectory_schema=job.output.trajectory_schema,
        )
        assert_record_sanity(record, scene, case)
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
