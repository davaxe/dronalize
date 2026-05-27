from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from dronalize.config import RuntimeOverride
from dronalize.core.errors import (
    ConfigurationError,
    DatasetNotFoundError,
    UnsupportedStorageBackendError,
)
from dronalize.datasets import (
    DatasetFeatureSupport,
    DatasetTemporalSupport,
    DatasetWindowingSupport,
    FrameBounds,
)
from dronalize.io import StorageBackend, read_manifest
from dronalize.io.backends.null import NullWriter
from dronalize.io.backends.registry import register_writer_backend
from dronalize.io.readers import PickleReader
from dronalize.runtime import (
    ExecutionRequest,
    OutputSample,
    execute_request,
    resolve_request,
    stream_plan,
)
from tests.support import DemoOptions, demo_descriptor

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.datasets import DatasetDescriptor
    from dronalize.io.records import SceneRecord


def _request(tmp_path: Path, **kwargs: object) -> ExecutionRequest:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    base: dict[str, object] = {
        "dataset": "demo",
        "input_dir": input_dir,
        "output_dir": output_dir,
        "storage_backend": StorageBackend.NULL,
    }
    base.update(kwargs)
    return ExecutionRequest.model_validate(base)


def _patch_descriptor(monkeypatch: pytest.MonkeyPatch, descriptor: DatasetDescriptor) -> None:
    def provider(_name: str) -> DatasetDescriptor:
        return descriptor

    monkeypatch.setattr("dronalize.runtime.api.get_dataset", provider)


def _patch_get_demo_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_descriptor(monkeypatch, demo_descriptor())


def test_resolve_request_builds_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    plan = resolve_request(_request(tmp_path, include_map=False))

    assert plan.dataset == "demo"
    assert plan.storage_backend == StorageBackend.NULL
    dataset_options = cast("DemoOptions", plan.loader.loader_options)
    assert dataset_options.batch_size == 2
    assert plan.map is None
    assert plan.effective_horizon_frames == 3
    assert plan.effective_default_observation_length == 2


def test_resolve_request_rejects_unknown_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    with pytest.raises(UnsupportedStorageBackendError, match="Unsupported storage backend"):
        _ = resolve_request(_request(tmp_path, storage_backend="bad-backend"))


def test_resolve_request_accepts_registered_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)
    register_writer_backend("test-null", lambda _plan: NullWriter.as_factory())

    plan = resolve_request(_request(tmp_path, storage_backend="test-null"))

    assert plan.storage_backend == "test-null"


def test_resolve_request_rejects_lane_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)
    config_path = tmp_path / "dronalize.toml"
    _ = config_path.write_text(
        """
[datasets.demo.scenes.lane_change]
persist = 3
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="does not support lane-change sampling"):
        _ = resolve_request(_request(tmp_path, config_path=config_path))


def test_resolve_request_requires_window_for_lane_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = replace(
        demo_descriptor(),
        feature_support=DatasetFeatureSupport(map=True, lane_change_sampling=True),
    )
    _patch_descriptor(monkeypatch, descriptor)
    config_path = tmp_path / "dronalize.toml"
    _ = config_path.write_text(
        """
[datasets.demo.scenes]
window = false

[datasets.demo.scenes.lane_change]
persist = 3
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="requires window sampling"):
        _ = resolve_request(_request(tmp_path, config_path=config_path))


def test_resolve_request_rejects_long_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = replace(
        demo_descriptor(),
        temporal_support=DatasetTemporalSupport(
            source_unit="scene",
            source_frame_bounds=FrameBounds(max_frames=2, confidence="documented"),
            windowing=DatasetWindowingSupport(enabled_by_default=True),
        ),
    )
    _patch_descriptor(monkeypatch, descriptor)

    with pytest.raises(ConfigurationError, match="supports windows up to 2 source frames"):
        _ = resolve_request(_request(tmp_path))


def test_resolve_request_rejects_window_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = replace(
        demo_descriptor(),
        temporal_support=DatasetTemporalSupport(
            source_unit="scene",
            source_frame_bounds=FrameBounds(max_frames=10, confidence="documented"),
            windowing=DatasetWindowingSupport(
                enabled_by_default=True, supported_policies=("strict",)
            ),
        ),
    )
    _patch_descriptor(monkeypatch, descriptor)
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(
        """
[datasets.demo.scenes.window]
step = 1
policy = "partial"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="does not support window policy 'partial'"):
        _ = resolve_request(_request(tmp_path, config_path=config_path))


def test_execute_request_surfaces_unknown_dataset(tmp_path: Path) -> None:
    request = _request(tmp_path, dataset="this-dataset-does-not-exist")

    with pytest.raises(DatasetNotFoundError):
        _ = execute_request(request)


def test_execute_request_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    request = _request(tmp_path)
    result = execute_request(request)

    assert result.dataset == "demo"
    assert result.storage_backend == StorageBackend.NULL
    assert result.processed_sources == 1

    manifest = read_manifest(result.output_dir)
    assert manifest.storage_backend == "null"
    assert manifest.source_trajectory_schema_fields == (
        "frame",
        "id",
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "yaw",
        "agent_category",
    )
    assert manifest.horizon_frames == 3
    assert manifest.default_observation_length == 2


def test_execute_request_applies_record_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    def transform(record: SceneRecord) -> dict[str, object]:
        return {
            "scene_number": record.scene_number,
            "dataset": record.dataset,
            "feature_shape": record.features.shape,
        }

    output_sample = OutputSample(record_transform=transform)
    request = _request(tmp_path, storage_backend=StorageBackend.PICKLE, output_sample=output_sample)

    result = execute_request(request)
    sample = cast("dict[str, object]", PickleReader(result.output_dir, sample_type=dict)[0])

    assert sample == {"scene_number": 0, "dataset": "demo", "feature_shape": (1, 3, 7)}


def test_execute_request_writes_custom_mds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip(
        "streaming", reason="Requires streaming package for custom MDS output sample format"
    )
    from dronalize.io.readers import MDSReader  # noqa: PLC0415

    _patch_get_demo_descriptor(monkeypatch)

    def transform(record: SceneRecord) -> dict[str, object]:
        return {
            "scene_number": record.scene_number,
            "dataset": record.dataset,
            "feature_shape": record.features.shape,
        }

    output_sample = OutputSample(
        record_transform=transform,
        mds_columns={"scene_number": "int", "dataset": "str", "feature_shape": "json"},
    )
    request = _request(tmp_path, storage_backend=StorageBackend.MDS, output_sample=output_sample)
    result = execute_request(request)
    reader = MDSReader(path=result.output_dir, convert_raw=dict)
    sample = reader[0]
    assert sample == {"scene_number": 0, "dataset": "demo", "feature_shape": [1, 3, 7]}


def test_parallel_execution_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    mp.set_start_method("spawn", force=True)
    _patch_get_demo_descriptor(monkeypatch)

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    request = ExecutionRequest(
        dataset="demo",
        input_dir=input_dir,
        output_dir=output_dir,
        storage_backend=StorageBackend.NULL,
        overrides=RuntimeOverride.from_inputs(jobs=2),
    )

    result = execute_request(request)

    assert result.dataset == "demo"
    assert result.processed_sources == 1
    assert result.candidate_scenes == 1
    assert result.selected_scenes == 1
    assert result.split_counts["unsplit"] == 1

    manifest = read_manifest(output_dir)
    assert manifest.horizon_frames == 3
    assert manifest.default_observation_length == 2


def test_parallel_stream_plan_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mp.set_start_method("spawn", force=True)
    _patch_get_demo_descriptor(monkeypatch)

    request = ExecutionRequest(
        dataset="demo",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        storage_backend=StorageBackend.NULL,
        overrides=RuntimeOverride.from_inputs(jobs=2),
    )
    request.input_dir.mkdir()

    plan = resolve_request(request)
    scenes = list(stream_plan(plan))

    assert len(scenes) == 1
    assert scenes[0].scene_number == 0
    assert scenes[0].dataset == "demo"
    assert scenes[0].has_map()
    assert scenes[0].resolve_map() is None
