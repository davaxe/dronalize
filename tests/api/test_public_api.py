from __future__ import annotations

import dronalize
from dronalize import datasets, io, processing, runtime
from dronalize.config import ProcessingConfig, RuntimeOverride, load_project_config
from dronalize.core import AgentCategory, DatasetSplit, EdgeType
from dronalize.core.maps import MapGraph, SharedMapGraph
from dronalize.io import DatasetManifest, manifest_path, read_manifest
from dronalize.io.base import DatasetWriter
from dronalize.io.manifest import write_manifest
from dronalize.runtime import (
    PlanningRequest,
    ProcessRequest,
    ProcessResult,
    RunPlan,
    process_dataset,
    resolve_job,
)


def test_root_namespace_is_intentionally_small() -> None:
    assert dronalize.__all__ == []


def test_main_namespaces_are_exposed() -> None:
    assert datasets is not None
    assert runtime is not None
    assert processing is not None
    assert io is not None


def test_core_and_runtime_exports_are_present() -> None:
    assert AgentCategory is not None
    assert DatasetSplit is not None
    assert EdgeType is not None
    assert MapGraph is not None
    assert SharedMapGraph is not None
    assert PlanningRequest is not None
    assert ProcessRequest is not None
    assert ProcessResult is not None
    assert RunPlan is not None
    assert resolve_job is not None
    assert process_dataset is not None


def test_io_and_config_exports_are_present() -> None:
    assert DatasetManifest is not None
    assert DatasetWriter is not None
    assert manifest_path is not None
    assert read_manifest is not None
    assert write_manifest is not None
    assert ProcessingConfig is not None
    assert RuntimeOverride is not None
    assert load_project_config is not None


def test_removed_runtime_executors_remain_internal() -> None:
    runtime_module = __import__("dronalize.runtime", fromlist=["ParallelExecutor"])
    assert not hasattr(runtime_module, "ParallelExecutor")
    assert not hasattr(runtime_module, "SequentialExecutor")
