from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from dronalize.config.runtime import RuntimeOverride
from dronalize.core.errors import (
    ConfigurationError,
    DatasetNotFoundError,
    UnsupportedStorageBackendError,
)
from dronalize.datasets import DatasetFeatureSupport
from dronalize.io import StorageBackend, read_manifest
from dronalize.io.backends.null import NullWriter
from dronalize.io.backends.registry import register_writer_backend
from dronalize.runtime import ExecutionRequest, execute_request, resolve_request, stream_plan
from tests.support import DemoOptions, demo_descriptor

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.datasets import DatasetDescriptor


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


def test_resolve_request_compiles_runtime_and_loader_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    plan = resolve_request(_request(tmp_path, include_map=False))

    assert plan.dataset == "demo"
    assert plan.storage_backend == StorageBackend.NULL
    dataset_options = cast("DemoOptions", plan.loader.loader_options)
    assert dataset_options.batch_size == 2
    assert plan.map is None
    assert plan.effective_history_frames == 2
    assert plan.effective_future_frames == 1


def test_resolve_request_rejects_unknown_storage_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    with pytest.raises(UnsupportedStorageBackendError, match="Unsupported storage backend"):
        _ = resolve_request(_request(tmp_path, storage_backend="bad-backend"))


def test_resolve_request_accepts_registered_string_storage_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)
    register_writer_backend("test-null", lambda _plan: NullWriter.as_factory())

    plan = resolve_request(_request(tmp_path, storage_backend="test-null"))

    assert plan.storage_backend == "test-null"


def test_resolve_request_rejects_unsupported_lane_change_sampling(
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


def test_resolve_request_rejects_lane_change_without_window(
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


def test_execute_request_surfaces_unknown_dataset_errors(tmp_path: Path) -> None:
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
    assert manifest.history_frames == 2
    assert manifest.future_frames == 1


def test_parallel_execution_smoke_processes_demo_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

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
    assert manifest.history_frames == 2
    assert manifest.future_frames == 1


def test_parallel_stream_plan_smoke_yields_demo_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert scenes[0].has_map()
    assert scenes[0].resolve_map() is None
