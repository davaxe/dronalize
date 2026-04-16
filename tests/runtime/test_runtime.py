from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from dronalize.core.errors import DatasetNotFoundError, UnsupportedStorageBackendError
from dronalize.io import StorageBackend, read_manifest
from dronalize.runtime import ExecutionRequest, execute_request, resolve_request
from tests.support import DemoOptions, demo_descriptor

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.datasets import DatasetSpec


def _request(tmp_path: Path, **kwargs: object) -> ExecutionRequest:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    base: dict[str, str | Path | StorageBackend | bool] = {
        "dataset": "demo",
        "input_dir": input_dir,
        "output_dir": output_dir,
        "storage_backend": StorageBackend.NULL,
    }
    base.update(cast("dict[str, str | Path | StorageBackend | bool]", kwargs))
    return ExecutionRequest.model_validate(base)


def _patch_get_demo_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor: DatasetSpec = demo_descriptor()

    def provider(_name: str) -> DatasetSpec:
        return descriptor

    monkeypatch.setattr("dronalize.runtime.api.get", provider)


def test_resolve_job_compiles_runtime_and_loader_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    plan = resolve_request(_request(tmp_path, include_map=False))

    assert plan.dataset == "demo"
    assert plan.storage_backend == StorageBackend.NULL
    dataset_options = cast("DemoOptions", plan.loader.dataset)
    assert dataset_options.batch_size == 2
    assert plan.map is None
    assert plan.effective_history_frames == 2
    assert plan.effective_future_frames == 1


def test_resolve_job_rejects_unknown_storage_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    with pytest.raises(UnsupportedStorageBackendError, match="Unsupported storage backend"):
        _ = resolve_request(_request(tmp_path, storage_backend="bad-backend"))


def test_process_dataset_surfaces_unknown_dataset_errors(tmp_path: Path) -> None:
    request = _request(tmp_path, dataset="this-dataset-does-not-exist")

    with pytest.raises(DatasetNotFoundError):
        _ = execute_request(request)


def test_run_job_executes_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get_demo_descriptor(monkeypatch)

    request = _request(tmp_path)
    result = execute_request(request)

    assert result.dataset == "demo"
    assert result.storage_backend == StorageBackend.NULL
    assert result.processed_sources == 1

    manifest = read_manifest(result.output_dir)
    assert manifest.history_frames == 2
    assert manifest.future_frames == 1
