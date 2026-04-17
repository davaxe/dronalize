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
- split strategies via [`SplitConfig`][dronalize.config.models.SplitConfig]
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
    MapExtraction,
    PartialMapConfig,
    SceneExtentExtraction,
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
    Derivatives,
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
    AgentSelector,
    CategoryRangeSpec,
    CleanupSpec,
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
    PruneByRuleSpec,
    Range,
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
    NativeSplitConfig,
    NoSplitConfig,
    SceneSplitConfig,
    ShuffledTimeSplitConfig,
    SourceSplitConfig,
    SplitConfig,
    SplitConfigUnion,
    SplitWeights,
    TimeSplitConfig,
)

__all__ = [
    "AgentCheckSpec",
    "AgentRangeSpec",
    "AgentSelector",
    "BoundingBoxExtraction",
    "CategoryRangeSpec",
    "CircularExtraction",
    "CleanupSpec",
    "DatasetConfig",
    "Derivatives",
    "EndsAfterFrameSpec",
    "ExcludeCategoriesSpec",
    "FullMapExtraction",
    "IncludeCategoriesSpec",
    "LaneChangeConfig",
    "MDSOutputConfig",
    "MapConfig",
    "MapExtraction",
    "MaxGapSpec",
    "MaxMissingFramesSpec",
    "MaxMissingSceneFramesSpec",
    "MinConsecutiveFramesSpec",
    "MinSamplesSpec",
    "MinSpanSpec",
    "NativeSplitConfig",
    "NoSplitConfig",
    "OutputConfig",
    "OutputPrecision",
    "PartialDatasetConfig",
    "PartialDatasetConfigBase",
    "PartialLaneChangeConfig",
    "PartialMDSOutputConfig",
    "PartialMapConfig",
    "PartialOutputConfig",
    "PartialResampleConfig",
    "PartialRuntimeConfig",
    "PartialScenesConfig",
    "PartialScreeningConfig",
    "PartialWindowConfig",
    "PruneByRuleSpec",
    "Range",
    "RequireFramesSpec",
    "RequireSceneFramesSpec",
    "RequireSceneWindowSpec",
    "RequireWindowSpec",
    "ResampleConfig",
    "RuntimeConfig",
    "SceneCheckSpec",
    "SceneExtentExtraction",
    "SceneSplitConfig",
    "ScenesConfig",
    "ScreeningConfig",
    "ShuffledTimeSplitConfig",
    "SourceSplitConfig",
    "SplitConfig",
    "SplitConfigUnion",
    "SplitWeights",
    "StartsByFrameSpec",
    "TimeSplitConfig",
    "Tolerance",
    "TrajectorySchemaLike",
    "WindowConfig",
    "effective_scene_window",
]
