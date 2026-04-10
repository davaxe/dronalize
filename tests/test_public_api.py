from __future__ import annotations

import dronalize
from dronalize import datasets, io, processing, runtime
from dronalize.core import AgentCategory, DatasetSplit, EdgeType
from dronalize.io import (
    DatasetManifest,
    DatasetWriter,
    manifest_path,
    read_manifest,
    write_manifest,
)
from dronalize.runtime import (
    PlanningRequest,
    ProcessRequest,
    ProcessResult,
    RunPlan,
    process_dataset,
    resolve_job,
)


def test_keep_top_namespace_small() -> None:
    """Keep root namespace minimal."""
    assert hasattr(dronalize, "__all__")
    assert dronalize.__all__ == []


def test_expose_package_namespaces() -> None:
    """Expose package namespaces."""
    assert datasets is not None
    assert runtime is not None
    assert processing is not None
    assert io is not None


def test_expose_core_enums() -> None:
    """Expose shared core enums."""
    assert AgentCategory is not None
    assert DatasetSplit is not None
    assert EdgeType is not None


def test_expose_runtime_api() -> None:
    """Expose runtime entrypoints."""
    assert PlanningRequest is not None
    assert ProcessRequest is not None
    assert ProcessResult is not None
    assert RunPlan is not None
    assert resolve_job is not None
    assert process_dataset is not None


def test_expose_io_api() -> None:
    """Expose IO contracts."""
    assert DatasetManifest is not None
    assert DatasetWriter is not None
    assert manifest_path is not None
    assert read_manifest is not None
    assert write_manifest is not None


def test_keep_removed_exports_absent() -> None:
    """Keep removed exports absent."""
    assert not hasattr(dronalize, "DatasetPlan")
    assert not hasattr(dronalize, "plan_dataset")
    runtime_module = __import__("dronalize.runtime", fromlist=["ParallelExecutor"])
    assert not hasattr(runtime_module, "ParallelExecutor")
    assert not hasattr(runtime_module, "SequentialExecutor")


def test_keep_registry_surface() -> None:
    """Keep dataset registry surface."""
    names = datasets.available()
    assert isinstance(names, list)
    assert "apolloscape" in names
