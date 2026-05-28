from __future__ import annotations

import dronalize
from dronalize import datasets, io, processing, runtime
from dronalize.config import ProjectConfig, RuntimeOverride, parse_config
from dronalize.core import AgentCategory, DatasetSplit, EdgeType
from dronalize.core.maps import MapGraph, SharedMapGraph
from dronalize.io import (
    DatasetManifest,
    SplitSceneRecord,
    manifest_path,
    read_manifest,
    write_manifest,
)
from dronalize.io import adapters as io_adapters
from dronalize.io import readers as io_readers
from dronalize.io.base import DatasetWriter
from dronalize.runtime import (
    ExecutionPlan,
    ExecutionRequest,
    ExecutionResult,
    OutputSample,
    execute_plan,
    execute_request,
    resolve_request,
)


def test_root_namespace_is_small() -> None:
    assert dronalize.__all__ == []


def test_main_namespaces_exposed() -> None:
    assert datasets is not None
    assert runtime is not None
    assert processing is not None
    assert io is not None
    assert datasets.DatasetTemporalSupport is not None
    assert datasets.DatasetWindowingSupport is not None
    assert datasets.FrameBounds is not None


def test_core_and_runtime_exports_present() -> None:
    assert AgentCategory is not None
    assert DatasetSplit is not None
    assert EdgeType is not None
    assert MapGraph is not None
    assert SharedMapGraph is not None
    assert ExecutionPlan is not None
    assert ExecutionRequest is not None
    assert ExecutionResult is not None
    assert OutputSample is not None
    assert execute_plan is not None
    assert resolve_request is not None
    assert execute_request is not None


def test_io_and_config_exports_present() -> None:
    assert DatasetManifest is not None
    assert SplitSceneRecord is not None
    assert DatasetWriter is not None
    assert manifest_path is not None
    assert read_manifest is not None
    assert write_manifest is not None
    assert ProjectConfig is not None
    assert RuntimeOverride is not None
    assert parse_config is not None


def test_reader_and_adapter_exports_declared() -> None:
    assert DatasetWriter is not None
    assert io_readers.DatasetReader is not None
    assert "DatasetReader" in io_readers.__all__
    assert "MDSReaderInitArgs" in io_readers.__all__
    assert "IterableTorchSceneDataset" in io_adapters.__all__
    assert "TorchSplitSceneDataset" in io_adapters.__all__
    assert "IterableHeteroSceneDataset" in io_adapters.__all__
    assert "SplitHeteroSceneDataset" in io_adapters.__all__


def test_runtime_executors_not_root_exports() -> None:
    runtime_module = __import__("dronalize.runtime", fromlist=["ParallelExecutor"])
    assert not hasattr(runtime_module, "ParallelExecutor")
    assert not hasattr(runtime_module, "SequentialExecutor")
