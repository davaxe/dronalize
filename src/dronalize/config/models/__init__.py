"""Typed configuration models grouped by processing concern.

Import from this package when you need to build, validate, or introspect
individual configuration sections directly in Python. The models are organized
roughly the same way as the project configuration structure:

- dataset-level composition via [`DatasetConfig`][dronalize.config.models.DatasetConfig]
- scene sampling and temporal transforms via
  [`ScenesConfig`][dronalize.config.models.ScenesConfig]
- screening and cleanup rules via
  [`ScreeningConfig`][dronalize.config.models.ScreeningConfig]
- output encoding and storage options via
  [`OutputConfig`][dronalize.config.models.OutputConfig]
- runtime execution controls via
  [`RuntimeConfig`][dronalize.config.models.RuntimeConfig]
- read selection via [`ReadConfig`][dronalize.config.models.ReadConfig]
- assignment strategies via [`AssignConfig`][dronalize.config.models.AssignConfig]
"""

from dronalize.config.models.dataset import (
    DatasetConfig,
    PartialDatasetConfig,
    PartialDatasetConfigBase,
)
from dronalize.config.models.map import (
    BoundingBoxExtraction,
    CircularExtraction,
    FullMapExtraction,
    MapConfig,
    MapEdgeTypeRules,
    MapExtraction,
    PartialMapConfig,
    PartialMapEdgeTypeRules,
    SceneExtentExtraction,
    TrajectoryBufferExtraction,
)
from dronalize.config.models.output import (
    MDSOutputConfig,
    OutputConfig,
    OutputPrecision,
    PartialMDSOutputConfig,
    PartialOutputConfig,
    TrajectorySchemaLike,
)
from dronalize.config.models.runtime import PartialRuntimeConfig, RuntimeConfig
from dronalize.config.models.scenes import (
    LaneChangeConfig,
    PartialLaneChangeConfig,
    PartialResampleConfig,
    PartialScenesConfig,
    PartialWindowConfig,
    ResampleConfig,
    ScenesConfig,
    WindowConfig,
    effective_scene_window,
)
from dronalize.config.models.screening import (
    AgentCheckSpec,
    AgentRangeSpec,
    AgentSelectorConfig,
    CategoryRangeSpec,
    CleanupSpec,
    CountRange,
    EndsAfterFrameSpec,
    ExcludeCategoriesSpec,
    IncludeCategoriesSpec,
    MaxGapSpec,
    MaxMissingFramesSpec,
    MaxMissingSceneFramesSpec,
    MinConsecutiveFramesSpec,
    MinSamplesSpec,
    MinSpanSpec,
    PartialScreeningConfig,
    PassingRequirement,
    PruneByRuleSpec,
    RequireFramesSpec,
    RequireSceneFramesSpec,
    RequireSceneWindowSpec,
    RequireWindowSpec,
    SceneCheckSpec,
    ScreeningConfig,
    StartsByFrameSpec,
    Tolerance,
)
from dronalize.config.models.split import (
    AssignConfig,
    AssignUnion,
    NoAssign,
    PreserveNativeAssign,
    ReadAll,
    ReadConfig,
    ReadNative,
    ReadUnion,
    SceneAssign,
    ShuffledTimeBlockAssign,
    SourceAssign,
    SplitWeights,
    TimeBlockAssign,
)

__all__ = [
    "AgentCheckSpec",
    "AgentRangeSpec",
    "AgentSelectorConfig",
    "AssignConfig",
    "AssignUnion",
    "BoundingBoxExtraction",
    "CategoryRangeSpec",
    "CircularExtraction",
    "CleanupSpec",
    "CountRange",
    "DatasetConfig",
    "EndsAfterFrameSpec",
    "ExcludeCategoriesSpec",
    "FullMapExtraction",
    "IncludeCategoriesSpec",
    "LaneChangeConfig",
    "MDSOutputConfig",
    "MapConfig",
    "MapEdgeTypeRules",
    "MapExtraction",
    "MaxGapSpec",
    "MaxMissingFramesSpec",
    "MaxMissingSceneFramesSpec",
    "MinConsecutiveFramesSpec",
    "MinSamplesSpec",
    "MinSpanSpec",
    "NoAssign",
    "OutputConfig",
    "OutputPrecision",
    "PartialDatasetConfig",
    "PartialDatasetConfigBase",
    "PartialLaneChangeConfig",
    "PartialMDSOutputConfig",
    "PartialMapConfig",
    "PartialMapEdgeTypeRules",
    "PartialOutputConfig",
    "PartialResampleConfig",
    "PartialRuntimeConfig",
    "PartialScenesConfig",
    "PartialScreeningConfig",
    "PartialWindowConfig",
    "PassingRequirement",
    "PreserveNativeAssign",
    "PruneByRuleSpec",
    "ReadAll",
    "ReadConfig",
    "ReadNative",
    "ReadUnion",
    "RequireFramesSpec",
    "RequireSceneFramesSpec",
    "RequireSceneWindowSpec",
    "RequireWindowSpec",
    "ResampleConfig",
    "RuntimeConfig",
    "SceneAssign",
    "SceneCheckSpec",
    "SceneExtentExtraction",
    "ScenesConfig",
    "ScreeningConfig",
    "ShuffledTimeBlockAssign",
    "SourceAssign",
    "SplitWeights",
    "StartsByFrameSpec",
    "TimeBlockAssign",
    "Tolerance",
    "TrajectoryBufferExtraction",
    "TrajectorySchemaLike",
    "WindowConfig",
    "effective_scene_window",
]
