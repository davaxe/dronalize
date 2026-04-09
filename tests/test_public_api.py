from __future__ import annotations

import dronalize
from dronalize.core import AgentCategory, DatasetSplit
from dronalize.core.maps import EdgeType, MapGraph, SharedMapGraph
from dronalize.core.scene import CANONICAL, Scene, TrajectorySchema
from dronalize.datasets import DatasetCapabilities, DatasetSpec, RuntimeContext, available, get, register
from dronalize.io import DatasetManifest, DatasetWriter, ExportConfig, MDSBackendConfig
from dronalize.io import adapters as io_adapters
from dronalize.io import readers as io_readers
from dronalize import plot
from dronalize.processing import LaneChangeSamplingConfig, LoaderConfig, MapConfig, WindowConfig
from dronalize.processing import filtering, maps, pipeline
from dronalize.processing.filtering import (
    AbsoluteTolerance,
    AgentSelector,
    CombinedTolerance,
    Filter,
    FilterSpec,
    RelativeTolerance,
    Tolerance,
    filter_scene,
    merge_filters,
    tol,
)
from dronalize.runtime import (
    ConfigFile,
    ConfigResolver,
    DatasetPlan,
    DatasetRun,
    FileDatasetConfig,
    FileExecutionConfig,
    PlanOverrides,
    ProcessingSummary,
    ResolvedConfig,
    ResolvedExecutionConfig,
    load_project_config,
    plan_dataset,
    resolve_runtime_config,
    summarize_plan,
)


def test_canonical_import_surfaces_are_available() -> None:
    """Canonical package surfaces should remain importable from their public modules."""
    assert AgentCategory is not None
    assert DatasetSplit is not None
    assert CANONICAL is not None
    assert Scene is not None
    assert TrajectorySchema is not None
    assert EdgeType is not None
    assert MapGraph is not None
    assert SharedMapGraph is not None
    assert DatasetCapabilities is not None
    assert DatasetSpec is not None
    assert RuntimeContext is not None
    assert available is not None
    assert get is not None
    assert register is not None
    assert ConfigFile is not None
    assert ConfigResolver is not None
    assert DatasetPlan is not None
    assert DatasetRun is not None
    assert FileDatasetConfig is not None
    assert FileExecutionConfig is not None
    assert PlanOverrides is not None
    assert ProcessingSummary is not None
    assert ResolvedConfig is not None
    assert ResolvedExecutionConfig is not None
    assert load_project_config is not None
    assert plan_dataset is not None
    assert resolve_runtime_config is not None
    assert summarize_plan is not None
    assert ExportConfig is not None
    assert MDSBackendConfig is not None
    assert DatasetManifest is not None
    assert DatasetWriter is not None
    assert io_readers.__all__ == ["MDSReader", "MDSReaderInitArgs"]
    assert io_adapters.__all__ == [
        "MDSHeteroDataset",
        "MDSTorchDataset",
        "TorchSceneRecord",
        "collate_hetero_with_time_padding",
    ]
    assert plot.__all__ == ["plot_map_graph", "plot_trajectories", "plot_trajectories_on_map"]
    assert LoaderConfig is not None
    assert WindowConfig is not None
    assert LaneChangeSamplingConfig is not None
    assert MapConfig is not None
    assert filtering is not None
    assert maps is not None
    assert pipeline is not None
    assert AgentSelector is not None
    assert AbsoluteTolerance is not None
    assert RelativeTolerance is not None
    assert CombinedTolerance is not None
    assert Tolerance is not None
    assert Filter is not None
    assert FilterSpec is not None
    assert filter_scene is not None
    assert merge_filters is not None
    assert tol is not None


def test_removed_convenience_exports_stay_absent() -> None:
    """Root and package shortcuts removed by the API cleanup should stay absent."""
    assert not hasattr(dronalize, "DatasetPlan")
    assert not hasattr(dronalize, "plan_dataset")
    assert not hasattr(__import__("dronalize.runtime", fromlist=["Executor"]), "Executor")
    assert not hasattr(__import__("dronalize.runtime", fromlist=["Progress"]), "Progress")
    assert not hasattr(__import__("dronalize.core.models", fromlist=["tol"]), "tol")
