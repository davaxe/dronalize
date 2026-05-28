"""Configuration models for dataset generation."""

from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar

import numpy as np
from pydantic import (
    AliasChoices,
    BeforeValidator,
    Field,
    RootModel,
    field_validator,
    model_validator,
)
from typing_extensions import override

from dronalize.config.base import (
    ConfigBase,
    ConfigPatch,
    ResampleMethod,
    ResolvedConfig,
    apply_optional,
)
from dronalize.core.categories import (
    AgentCategoryLike,
    DatasetSplit,
    EdgeType,
    EdgeTypeLike,
    coerce_edge_types,
)
from dronalize.core.errors import ConfigurationError
from dronalize.core.functional.window import WindowPolicy  # noqa: TC001
from dronalize.core.scene import CANONICAL, TrajectorySchema
from dronalize.core.scene.schema import TrajectorySchemaDefinition

if TYPE_CHECKING:
    from dronalize.core.typing import T


Categories = tuple[AgentCategoryLike, ...]
"""Tuple of category selectors accepted by screening rules."""

EdgeTypes = Annotated[
    frozenset[EdgeType], BeforeValidator(lambda v: coerce_edge_types(v, frozenset))
]

FloatDType = type[np.float32] | type[np.float64]
OutputPrecision = Literal["float32", "float64"]
"""Accepted floating-point precision labels for serialized numeric arrays.

The chosen label is later resolved to the corresponding NumPy dtype when
writers materialize trajectory tensors.
"""

TrajectorySchemaLike = TrajectorySchema | str | TrajectorySchemaDefinition
"""User-facing trajectory schema inputs accepted by output-related config.

Callers may provide a registered schema object directly, reference a schema by
name, or inline a full schema definition that resolves to a
[`TrajectorySchema`][dronalize.core.scene.TrajectorySchema].
"""

ReadStrategy = Literal["all", "native"]
"""Supported runtime read-strategy override names accepted by the CLI layer.

The string values map directly to the concrete read config models in
[`dronalize.config.models`][].
"""

AssignStrategy = Literal["none", "preserve-native", "scene", "source", "time", "shuffled-time"]
"""Supported runtime assignment-strategy override names accepted by the CLI layer.

The string values map directly to the concrete assignment config models in
[`dronalize.config.models`][].
"""


class SplitWeights(ResolvedConfig):
    """Weights used when routing data into train/val/test assignments."""

    train: float = Field(ge=0, default=0.0)
    """Weight assigned to the training split."""
    val: float = Field(ge=0, default=0.0)
    """Weight assigned to the validation split."""
    test: float = Field(ge=0, default=0.0)
    """Weight assigned to the test split."""

    @model_validator(mode="after")
    def _validate_sum(self) -> SplitWeights:
        """Validate that at least one split weight is positive."""
        total = self.train + self.val + self.test
        if total == 0:
            v = f"train={self.train}, val={self.val}, test={self.test}"
            msg = f"At least one split weight must be greater than 0, got {v}"
            raise ConfigurationError(msg)
        return self


class ReadAll(ResolvedConfig):
    """Read the full dataset input surface."""

    strategy: Literal["all"] = Field("all", repr=False, init=False)


class ReadNative(ResolvedConfig):
    """Read only selected dataset-native partitions."""

    strategy: Literal["native"] = Field("native", repr=False, init=False)
    splits: frozenset[DatasetSplit] | None = None
    """Native dataset partitions to read from the DatasetSource dataset."""


ReadUnion = Annotated[ReadAll | ReadNative, Field(discriminator="strategy")]


class ReadConfig(RootModel[ReadUnion]):
    """Read-selection configuration wrapper model."""

    root: ReadUnion


class NoAssign(ResolvedConfig):
    """No output split assignment."""

    strategy: Literal["none"] = Field("none", repr=False, init=False)


class PreserveNativeAssign(ResolvedConfig):
    """Preserve dataset-native split labels in output."""

    strategy: Literal["preserve-native"] = Field("preserve-native", repr=False, init=False)


class SceneAssign(ResolvedConfig):
    """Scene-based output split assignment."""

    strategy: Literal["scene"] = Field("scene", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


class SourceAssign(ResolvedConfig):
    """DatasetSource-based output split assignment."""

    strategy: Literal["source"] = Field("source", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


class TimeBlockAssign(ResolvedConfig):
    """Time-block output split assignment."""

    gap: int = Field(ge=0, default=0)
    strategy: Literal["time"] = Field("time", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


class ShuffledTimeBlockAssign(ResolvedConfig):
    """Shuffled time-block output split assignment."""

    segments: int = Field(ge=1)
    gap: int = Field(ge=0, default=0)
    strategy: Literal["shuffled-time"] = Field("shuffled-time", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


AssignUnion = Annotated[
    NoAssign
    | PreserveNativeAssign
    | SceneAssign
    | SourceAssign
    | TimeBlockAssign
    | ShuffledTimeBlockAssign,
    Field(discriminator="strategy"),
]


class AssignConfig(RootModel[AssignUnion]):
    """Assignment configuration wrapper model."""

    root: AssignUnion


class RuntimeConfig(ResolvedConfig):
    """Execution controls for processing requests.

    Attributes
    ----------
    jobs : int | Literal["auto"]
        Number of worker processes to use. Set to `"auto"` to let runtime
        choose an appropriate value.
    chunksize : int | None
        Optional per-worker batch size for scene dispatch.
    """

    jobs: int = Field(default=1, gt=0)
    """Number of worker processes to use."""
    chunksize: int | None = Field(default=None, gt=0)
    """Optional per-worker batch size for scene dispatch."""


class PartialRuntimeConfig(ConfigPatch[RuntimeConfig]):
    """Patch model for partially overriding :class:`RuntimeConfig`."""

    jobs: int | Literal["auto"] | None = None
    """Replacement worker count or `"auto"` to use the current CPU count."""
    chunksize: int | None = Field(default=None, gt=0)
    """Replacement per-worker batch size for scene dispatch."""
    full_config_type: type[RuntimeConfig] = Field(default=RuntimeConfig, init=False, repr=False)

    @override
    def merge_into(
        self, target: RuntimeConfig | None, *, exclude_none: bool = True
    ) -> RuntimeConfig:
        partial = self.model_copy(update={"jobs": mp.cpu_count()}) if self.jobs == "auto" else self
        return ConfigPatch[RuntimeConfig].merge_into(partial, target, exclude_none=exclude_none)


class MDSOutputConfig(ResolvedConfig):
    """Backend-specific tuning for Mosaic Streaming output."""

    compression: str | None = None
    """Compression algorithm to use for each Mosaic shard, if any."""
    hashes: tuple[str, ...] | None = None
    """Hash algorithms recorded for each shard, if hashing is enabled."""
    size_limit: str | int = 67_108_864
    """Maximum shard size accepted by the writer before starting a new shard."""
    exist_ok: bool = False
    """Whether an existing output location may be reused instead of raising an error."""


class PartialMDSOutputConfig(ConfigPatch[MDSOutputConfig]):
    """Patch model for partially overriding Mosaic Streaming writer settings."""

    compression: str | None = None
    """Replacement compression algorithm for Mosaic shards."""
    hashes: tuple[str, ...] | None = None
    """Replacement hash algorithms recorded for Mosaic shards."""
    size_limit: str | int | None = None
    """Replacement maximum shard size for the Mosaic writer."""
    exist_ok: bool | None = None
    """Replacement policy for whether existing output locations are allowed."""
    full_config_type: type[MDSOutputConfig] = Field(default=MDSOutputConfig, init=False, repr=False)


class OutputConfig(ResolvedConfig):
    """Resolved output configuration shared by storage backends."""

    trajectory_schema: TrajectorySchemaLike = Field(
        default=CANONICAL,
        validation_alias=AliasChoices("schema", "trajectory_schema"),
        serialization_alias="schema",
    )
    """Trajectory schema used when encoding scene records."""
    precision: OutputPrecision = "float32"
    """Floating-point precision used for serialized numeric arrays."""
    recenter_positions: bool = True
    """Whether scene positions are translated into a local origin before writing."""
    mds: MDSOutputConfig = Field(default_factory=MDSOutputConfig)
    """Backend-specific tuning for Mosaic Streaming outputs."""


class PartialOutputConfig(ConfigPatch[OutputConfig]):
    """Patch model for partially overriding shared output settings."""

    trajectory_schema: TrajectorySchemaLike | None = Field(
        default=None,
        validation_alias=AliasChoices("schema", "trajectory_schema"),
        serialization_alias="schema",
    )
    """Replacement trajectory schema used when encoding scene records."""
    precision: OutputPrecision | None = None
    """Replacement floating-point precision for serialized numeric arrays."""
    recenter_positions: bool | None = None
    """Replacement policy for recentering scene positions before writing."""
    mds: PartialMDSOutputConfig | None = None
    """Partial backend-specific overrides for Mosaic Streaming outputs."""
    full_config_type: type[OutputConfig] = Field(default=OutputConfig, init=False, repr=False)


class SceneExtentExtraction(ResolvedConfig):
    """Configuration for extraction around the scene trajectory extent."""

    mode: Literal["scene_extent"] = Field("scene_extent", repr=False, init=False)
    padding: float = Field(ge=1.0, default=1.0)
    """Scale factor applied around the scene extent before cropping the map."""
    shape: Literal["circle", "bounding_box"] = Field(default="circle")
    """Adaptive crop shape used around the scene extent."""


class CircularExtraction(ResolvedConfig):
    """Configuration for circular map extraction mode."""

    mode: Literal["circle"] = Field("circle", repr=False, init=False)
    radius: float = Field(gt=0)
    """Radius of the circular crop centered on the scene in map units."""


class BoundingBoxExtraction(ResolvedConfig):
    """Configuration for bounding-box map extraction mode."""

    mode: Literal["bounding_box"] = Field("bounding_box", repr=False, init=False)
    width: float = Field(gt=0)
    """Width of the extracted map crop in map units."""
    height: float = Field(gt=0)
    """Height of the extracted map crop in map units."""


class TrajectoryBufferExtraction(ResolvedConfig):
    """Configuration for buffering the scene trajectories directly."""

    mode: Literal["trajectory_buffer"] = Field("trajectory_buffer", repr=False, init=False)
    radius: float = Field(gt=0)
    """Buffer radius around each relevant trajectory point in map units."""


class FullMapExtraction(ResolvedConfig):
    """Configuration for keeping the full map without cropping."""

    mode: Literal["full"] = Field("full", repr=False, init=False)


class MapEdgeTypeRules(ResolvedConfig):
    """Semantic edge-type filtering and remapping rules."""

    include: EdgeTypes | None = Field(default=None)
    """Optional allow-list of edge types to keep after remapping."""
    exclude: EdgeTypes = frozenset()
    """Edge types to drop after remapping."""
    remap: dict[EdgeType, EdgeType] = Field(default_factory=dict)
    """Mapping applied to edge types before include/exclude filters."""

    @model_validator(mode="after")
    def _validate_no_conflicts(self) -> MapEdgeTypeRules:
        if self.include is not None and self.exclude.intersection(self.include):
            overlap = self.exclude.intersection(self.include)
            msg = f"Conflict in edge type rules: {overlap}."
            raise ValueError(msg)
        if self.remap.keys() & self.remap.values():
            overlap = self.remap.keys() & self.remap.values()
            msg = f"Conflict in edge type remapping: {overlap}."
            raise ValueError(msg)
        return self

    @field_validator("remap", mode="before")
    @classmethod
    def _coerce_remap_keys_and_values(
        cls, v: dict[EdgeTypeLike, EdgeTypeLike]
    ) -> dict[EdgeType, EdgeType]:
        return {EdgeType.from_value(k): EdgeType.from_value(val) for k, val in v.items()}


MapExtraction = Annotated[
    CircularExtraction
    | BoundingBoxExtraction
    | FullMapExtraction
    | SceneExtentExtraction
    | TrajectoryBufferExtraction,
    Field(discriminator="mode"),
]
"""Discriminated union of the supported map extraction strategies.

The `mode` field selects whether map geometry is cropped around the scene,
within a circle, within a bounding box, around the trajectory points directly,
or retained in full.
"""


class MapConfig(ResolvedConfig):
    """Configuration for map data processing."""

    min_distance: float | None = Field(gt=0, default=2)
    """Minimum spacing allowed between neighboring map samples after simplification."""
    interpolation_distance: float | None = Field(gt=0, default=5.0)
    """Target spacing used when interpolating map geometry."""
    extraction: MapExtraction = Field(default_factory=FullMapExtraction)
    """Map extraction strategy used to crop or retain DatasetSource map geometry."""
    edge_types: MapEdgeTypeRules | None = Field(default=None)
    """Optional edge-type remapping and filtering rules."""

    @model_validator(mode="after")
    def _validate_distances(self) -> MapConfig:
        if self.interpolation_distance is None or self.min_distance is None:
            return self
        if self.interpolation_distance < self.min_distance:
            msg = (
                f"interpolation_distance ({self.interpolation_distance}) must be greater "
                f"than or equal to min_distance ({self.min_distance})."
            )
            raise ValueError(msg)
        return self


class PartialMapEdgeTypeRules(ConfigPatch[MapEdgeTypeRules]):
    """Patch model for partially overriding :class:`MapEdgeTypeRules`."""

    include: frozenset[EdgeTypeLike] | None = Field(default=None)
    """Replacement allow-list of edge types to keep."""
    exclude: frozenset[EdgeTypeLike] | None = Field(default=None)
    """Replacement deny-list of edge types to drop."""
    remap: dict[EdgeTypeLike, EdgeTypeLike] | None = Field(default=None)
    """Replacement edge-type remapping rules."""
    full_config_type: type[MapEdgeTypeRules] = Field(MapEdgeTypeRules, repr=False, init=False)


class PartialMapConfig(ConfigPatch[MapConfig]):
    """Patch model for partially overriding :class:`MapConfig`."""

    min_distance: float | None = Field(gt=0, default=None)
    """Replacement minimum spacing for simplified map samples."""
    interpolation_distance: float | None = Field(gt=0, default=None)
    """Replacement interpolation spacing for map geometry."""
    extraction: MapExtraction | None = Field(default=None)
    """Replacement map extraction strategy."""
    edge_types: PartialMapEdgeTypeRules | Literal[False] | None = Field(default=None)
    """Replacement edge-type rules, or `false` to clear inherited rules."""
    full_config_type: type[MapConfig] = Field(MapConfig, repr=False, init=False)

    @override
    def merge_into(self, target: MapConfig | None, *, exclude_none: bool = True) -> MapConfig:
        if exclude_none is False:
            return super().merge_into(target, exclude_none=exclude_none)

        base = MapConfig() if target is None else target
        return MapConfig(
            min_distance=self.min_distance if self.min_distance is not None else base.min_distance,
            interpolation_distance=(
                self.interpolation_distance
                if self.interpolation_distance is not None
                else base.interpolation_distance
            ),
            extraction=self.extraction if self.extraction is not None else base.extraction,
            edge_types=apply_optional(self.edge_types, base.edge_types),
        )


class ResampleConfig(ResolvedConfig):
    """Validated specification for temporal resampling."""

    up: int = Field(default=1, gt=0)
    """Upsampling factor applied before downsampling."""
    down: int = Field(default=1, gt=0)
    """Downsampling factor applied after upsampling."""
    method: ResampleMethod = "linear"
    """Interpolation method used during resampling."""
    coordinates: tuple[str, ...] = Field(default=("x", "y"))
    """Coordinate fields resampled by the interpolation step."""
    emit_velocity: bool = Field(default=False)
    """Whether velocity derivatives should be emitted during resampling."""
    emit_acceleration: bool = Field(default=False)
    """Whether acceleration derivatives should be emitted during resampling."""
    max_gap: int = Field(default=1, gt=0)
    """Maximum consecutive missing frames allowed when interpolating samples."""

    @model_validator(mode="after")
    def _validate(self) -> ResampleConfig:
        if self.method == "linear" and (self.emit_velocity or self.emit_acceleration):
            msg = "Linear resampling does not support emitting derivatives."
            raise ConfigurationError(msg)
        return self


class PartialResampleConfig(ConfigPatch[ResampleConfig]):
    """Patch model for partially overriding temporal resampling settings."""

    up: int | None = None
    """Replacement upsampling factor."""
    down: int | None = None
    """Replacement downsampling factor."""
    method: ResampleMethod | None = None
    """Replacement interpolation method."""
    coordinates: tuple[str, ...] | None = None
    """Replacement coordinate fields to resample."""
    emit_velocity: bool | None = None
    """Replacement policy for emitting velocity derivatives."""
    emit_acceleration: bool | None = None
    """Replacement policy for emitting acceleration derivatives."""
    max_gap: int | None = None
    """Replacement maximum consecutive gap allowed during interpolation."""
    full_config_type: type[ResampleConfig] = Field(default=ResampleConfig, init=False, repr=False)


class WindowConfig(ResolvedConfig):
    """Configuration for sliding window sampling of scenes."""

    step: int = Field(gt=0)
    """Stride between consecutive sampled windows in frames."""
    policy: WindowPolicy = "strict"
    """Completeness policy for sources that do not fully cover a window."""


class PartialWindowConfig(ConfigPatch[WindowConfig]):
    """Patch model for partially overriding sliding-window sampling settings."""

    step: int | None = None
    """Replacement stride between consecutive sampled windows in frames."""
    policy: WindowPolicy | None = None
    """Replacement completeness policy for incomplete windows."""
    full_config_type: type[WindowConfig] = Field(default=WindowConfig, init=False, repr=False)


class LaneChangeConfig(ResolvedConfig):
    """Configuration for lane-change-aware sampling."""

    persist: int = Field(gt=0)
    """Minimum number of frames a lane-change state must persist to count."""
    margin_before: int = Field(default=0, ge=0)
    """Extra context frames to keep before the detected lane change."""
    margin_after: int = Field(default=0, ge=0)
    """Extra context frames to keep after the detected lane change."""
    required_lane_changes: int = Field(default=1, gt=0)
    """Minimum number of lane changes required for a positive sample."""
    negative_keep_every: int = Field(default=3, ge=1)
    """Keep one negative sample out of every N candidates."""


class PartialLaneChangeConfig(ConfigPatch[LaneChangeConfig]):
    """Patch model for partially overriding lane-change-aware sampling settings."""

    persist: int | None = None
    """Replacement persistence threshold for lane-change detection."""
    margin_before: int | None = None
    """Replacement context margin before a detected lane change."""
    margin_after: int | None = None
    """Replacement context margin after a detected lane change."""
    required_lane_changes: int | None = None
    """Replacement minimum lane-change count for positive samples."""
    negative_keep_every: int | None = None
    """Replacement negative-sample retention interval."""
    full_config_type: type[LaneChangeConfig] = Field(
        default=LaneChangeConfig, init=False, repr=False
    )


class ScenesConfig(ResolvedConfig):
    """Base configuration class for scene construction and temporal transforms."""

    horizon_frames: int = Field(gt=0)
    """Number of frames included in each scene horizon."""
    default_observation_length: int | None = Field(default=None, ge=0)
    """Optional default split point for reader/adaptor convenience.

    This is provided for datasets that have a natural split between observation
    and prediction frames, but it is not required for correct operation. If set,
    this value will be used as the default observation length for any reader or
    adaptor that supports a separate observation/prediction split, but it can be
    overridden at the reader or adaptor level if needed. If not set, readers and
    adaptors will need to be configured with an explicit observation length or
    callable to determine the split point.
    """
    sample_time: float = Field(gt=0)
    """Time interval between consecutive frames in seconds."""
    window: WindowConfig | None = Field(default=None)
    """Optional sliding-window sampling configuration."""
    resample: ResampleConfig | None = Field(default=None)
    """Optional temporal resampling configuration applied before scene emission."""
    lane_change: LaneChangeConfig | None = Field(default=None)
    """Optional lane-change-aware sampling configuration."""

    @model_validator(mode="after")
    def _validate(self) -> ScenesConfig:
        if (
            self.default_observation_length is not None
            and self.default_observation_length > self.horizon_frames
        ):
            msg = (
                "`default_observation_length` must be less than or equal to "
                f"`horizon_frames` ({self.horizon_frames})."
            )
            raise ConfigurationError(msg)
        return self


class PartialScenesConfig(ConfigPatch[ScenesConfig]):
    """Patch model for partially overriding scene construction settings."""

    horizon_frames: int | None = None
    """Replacement number of frames per scene horizon."""
    default_observation_length: int | None = None
    """Replacement default reader/adaptor split point."""
    sample_time: float | None = None
    """Replacement frame interval in seconds."""
    window: PartialWindowConfig | Literal[False] | None = None
    """Partial override for sliding-window sampling settings."""
    resample: PartialResampleConfig | Literal[False] | None = None
    """Partial override for temporal resampling settings."""
    lane_change: PartialLaneChangeConfig | Literal[False] | None = None
    """Partial override for lane-change-aware sampling settings."""
    full_config_type: type[ScenesConfig] = Field(default=ScenesConfig, init=False, repr=False)

    @override
    def merge_into(self, target: ScenesConfig | None, *, exclude_none: bool = True) -> ScenesConfig:
        """Apply this partial scenes config to an existing full scenes config."""
        return ScenesConfig(
            horizon_frames=_resolve_required(
                "horizon_frames",
                self.horizon_frames,
                target.horizon_frames if target is not None else None,
            ),
            default_observation_length=(
                self.default_observation_length
                if self.default_observation_length is not None
                else (target.default_observation_length if target is not None else None)
            ),
            sample_time=_resolve_required(
                "sample_time", self.sample_time, target.sample_time if target is not None else None
            ),
            window=_apply_optional_block(
                self.window, target.window if target is not None else None
            ),
            resample=_apply_optional_block(
                self.resample, target.resample if target is not None else None
            ),
            lane_change=_apply_optional_block(
                self.lane_change, target.lane_change if target is not None else None
            ),
        )


def _resolve_required(name: str, value: T | None, fallback: T | None) -> T:
    result = value if value is not None else fallback
    if result is None:
        msg = f"Missing required field: {name}"
        raise ValueError(msg)
    return result


ConfigT = TypeVar("ConfigT", bound=ConfigBase)


def _apply_optional_block(
    patch: ConfigPatch[ConfigT] | Literal[False] | None, target: ConfigT | None
) -> ConfigT | None:
    """Apply a patch to an optional nested config block."""
    if patch is None:
        return target
    if patch is False:
        return None
    return patch.merge_into(target)


def effective_scene_window(config: ScenesConfig) -> tuple[int, int | None, float]:
    """Return horizon frames, default observation length, and sample time after resampling."""
    if config.resample is None:
        return config.horizon_frames, config.default_observation_length, config.sample_time

    up = config.resample.up
    down = config.resample.down
    ratio = up / down
    horizon_resampled = _resample_length(config.horizon_frames, ratio)
    observation_resampled = (
        None
        if config.default_observation_length is None
        else _resample_length(config.default_observation_length, ratio)
    )
    return (horizon_resampled, observation_resampled, config.sample_time * down / up)


def _resample_length(length: int, ratio: float) -> int:
    if length <= 0:
        return 0
    return int((length - 1) * ratio + 1)


class AgentSelectorConfig(ResolvedConfig):
    """Category selection mode applied before evaluating a rule.

    `include` keeps only listed categories for the rule evaluation, while
    `exclude` applies the rule to all categories except the listed ones.
    """

    mode: Literal["include", "exclude"] = Field("include")
    """Whether listed categories are included in or excluded from the rule evaluation."""
    categories: Categories
    """Categories included in or excluded from the rule evaluation."""


class Tolerance(ResolvedConfig):
    """Optional tolerance thresholds for relaxed rule checks."""

    absolute: float | None = Field(default=None, gt=0.0)
    """Absolute tolerance applied when evaluating numeric thresholds."""
    relative: float | None = Field(default=None, gt=0.0, le=1.0)
    """Relative tolerance applied when evaluating numeric thresholds."""

    @model_validator(mode="after")
    def _require_tolerance(self) -> Tolerance:
        if self.absolute is None and self.relative is None:
            msg = "at least one of absolute or relative must be set."
            raise ValueError(msg)
        return self


class PassingRequirement(ResolvedConfig):
    """Minimum selected-agent pass thresholds for agent-rule scene acceptance."""

    absolute: int | None = Field(default=None, ge=1)
    """Absolute number of selected agents that must pass the rule."""
    relative: float | None = Field(default=None, gt=0.0, le=1.0)
    """Relative fraction of selected agents that must pass the rule."""

    @model_validator(mode="after")
    def _require_threshold(self) -> PassingRequirement:
        if self.absolute is None and self.relative is None:
            msg = "at least one of absolute or relative must be set."
            raise ValueError(msg)
        return self


class CountRange(ResolvedConfig):
    """Generic integer range with optional minimum and maximum."""

    minimum: int | None = None
    """Inclusive lower bound for the range, if one is required."""
    maximum: int | None = None
    """Inclusive upper bound for the range, if one is required."""

    @model_validator(mode="after")
    def _val_range(self) -> CountRange:
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            msg = "maximum must be greater than or equal to minimum."
            raise ValueError(msg)
        return self


class _AgentRuleSpecBase(ResolvedConfig):
    """Shared options for per-agent screening rules."""

    selector: AgentSelectorConfig | None = Field(default=None)
    """Optional category selector limiting which agents the rule evaluates."""
    tolerance: Tolerance | None = Field(default=None)
    """Optional tolerance thresholds used to relax rule comparisons."""
    require: PassingRequirement | None = Field(default=None)
    """Optional minimum selected-agent pass thresholds required to keep the scene."""


class MinDistanceSpec(_AgentRuleSpecBase):
    """Require a minimum distance between first and last observations."""

    rule: Literal["min_distance"] = Field("min_distance", repr=False, init=False)
    minimum: float = Field(ge=0.0)
    """Minimum distance between first and last observations for each selected agent."""


class RequireFramesSpec(_AgentRuleSpecBase):
    """Require listed frame IDs to exist for each selected agent."""

    rule: Literal["frames"] = Field("frames", repr=False, init=False)
    frames: tuple[int, ...]
    """Frame indices that must be present for each selected agent."""


class RequireWindowSpec(_AgentRuleSpecBase):
    """Require agent coverage within an inclusive frame window."""

    rule: Literal["window"] = Field("window", repr=False, init=False)
    start_frame: int = Field(ge=0)
    """Inclusive start frame of the required coverage window."""
    end_frame: int = Field(ge=0)
    """Inclusive end frame of the required coverage window."""
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    """Minimum fraction of frames within the window that must be present."""


class MinSamplesSpec(_AgentRuleSpecBase):
    """Require a minimum number of samples for each selected agent."""

    rule: Literal["min_samples"] = Field("min_samples", repr=False, init=False)
    minimum: int = Field(ge=1)
    """Minimum number of samples required for each selected agent."""


class MaxMissingFramesSpec(_AgentRuleSpecBase):
    """Limit how many frames may be missing per selected agent."""

    rule: Literal["max_missing_frames"] = Field("max_missing_frames", repr=False, init=False)
    maximum: int = Field(ge=0)
    """Maximum number of missing frames allowed for each selected agent."""


class MaxGapSpec(_AgentRuleSpecBase):
    """Limit the longest consecutive missing gap per selected agent."""

    rule: Literal["max_gap"] = Field("max_gap", repr=False, init=False)
    maximum: int = Field(ge=0)
    """Maximum consecutive missing-frame gap allowed for each selected agent."""


class MinConsecutiveFramesSpec(_AgentRuleSpecBase):
    """Require a minimum uninterrupted run of valid frames."""

    rule: Literal["min_consecutive_frames"] = Field(
        "min_consecutive_frames", repr=False, init=False
    )
    minimum: int = Field(ge=1)
    """Minimum uninterrupted run of valid frames required per selected agent."""


class StartsByFrameSpec(_AgentRuleSpecBase):
    """Require each selected agent to start at or before a frame index."""

    rule: Literal["starts_by_frame"] = Field("starts_by_frame", repr=False, init=False)
    frame: int = Field(ge=0)
    """Latest frame index by which each selected agent must have started."""


class EndsAfterFrameSpec(_AgentRuleSpecBase):
    """Require each selected agent to continue past a frame index."""

    rule: Literal["ends_after_frame"] = Field("ends_after_frame", repr=False, init=False)
    frame: int = Field(ge=0)
    """Frame index after which each selected agent must still be present."""


class MinSpanSpec(_AgentRuleSpecBase):
    """Require a minimum frame-span between first and last observations."""

    rule: Literal["min_span"] = Field("min_span", repr=False, init=False)
    minimum: int = Field(ge=1)
    """Minimum frame span between first and last observations for each selected agent."""


AgentCheckSpec = Annotated[
    RequireFramesSpec
    | RequireWindowSpec
    | MinSamplesSpec
    | MaxMissingFramesSpec
    | MaxGapSpec
    | MinConsecutiveFramesSpec
    | StartsByFrameSpec
    | EndsAfterFrameSpec
    | MinSpanSpec
    | MinDistanceSpec,
    Field(discriminator="rule"),
]
"""Discriminated union of all declarative per-agent screening rule models.

Each variant describes one rule family that can be placed in the `agent`
section of a [`ScreeningConfig`][dronalize.config.models.ScreeningConfig].
"""


class AgentRangeSpec(ResolvedConfig):
    """Constrain the count of selected agents present in a scene."""

    rule: Literal["agent_range"] = Field("agent_range", repr=False, init=False)
    selector: AgentSelectorConfig | None = None
    """Optional category selector limiting which agents are counted."""
    minimum: int | None = Field(default=None, ge=0)
    """Inclusive lower bound on the number of matching agents in a scene."""
    maximum: int | None = Field(default=None, ge=0)
    """Inclusive upper bound on the number of matching agents in a scene."""


class CategoryRangeSpec(ResolvedConfig):
    """Constrain per-category agent counts using named integer ranges."""

    rule: Literal["category_range"] = Field("category_range", repr=False, init=False)
    ranges: dict[str, CountRange]
    """Required count ranges keyed by category name."""


class RequireSceneFramesSpec(ResolvedConfig):
    """Require specific frame IDs to exist at the scene level."""

    rule: Literal["scene_frames"] = Field("scene_frames", repr=False, init=False)
    frames: tuple[int, ...]
    """Frame indices that must exist at scene scope."""


class RequireSceneWindowSpec(ResolvedConfig):
    """Require scene occupancy over an inclusive frame window."""

    rule: Literal["scene_window"] = Field("scene_window", repr=False, init=False)
    start_frame: int = Field(ge=0)
    """Inclusive start frame of the required scene coverage window."""
    end_frame: int = Field(ge=0)
    """Inclusive end frame of the required scene coverage window."""
    min_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    """Minimum fraction of scene frames within the window that must exist."""


class MaxMissingSceneFramesSpec(ResolvedConfig):
    """Limit missing frames at scene scope for a selected category set."""

    rule: Literal["max_missing_frames"] = Field("max_missing_frames", repr=False, init=False)
    selector: AgentSelectorConfig
    """Category selector defining which agents contribute to the scene-level check."""
    maximum: int = Field(ge=0)
    """Maximum number of missing scene frames allowed for the selected agents."""


SceneCheckSpec = Annotated[
    AgentRangeSpec
    | CategoryRangeSpec
    | RequireSceneFramesSpec
    | RequireSceneWindowSpec
    | MaxMissingSceneFramesSpec,
    Field(discriminator="rule"),
]
"""Discriminated union of all declarative scene-level screening rule models.

These models power the `scene` section of
[`ScreeningConfig`][dronalize.config.models.ScreeningConfig].
"""


class PruneByRuleSpec(ResolvedConfig):
    """Cleanup action that removes agents matching a nested agent rule."""

    rule: Literal["prune_by"] = Field("prune_by", repr=False, init=False)
    agent_rule: AgentCheckSpec
    """Nested agent rule used to remove matching agents from the scene."""

    @model_validator(mode="after")
    def _reject_aggregate_requirements(self) -> PruneByRuleSpec:
        if self.agent_rule.require is not None:
            msg = "`require` is only valid for agent screening rules, not cleanup pruning."
            raise ValueError(msg)
        return self


class ExcludeCategoriesSpec(ResolvedConfig):
    """Cleanup action that drops listed categories from a scene."""

    rule: Literal["exclude"] = Field("exclude", repr=False, init=False)
    categories: Categories
    """Categories removed from the scene during cleanup."""


class IncludeCategoriesSpec(ResolvedConfig):
    """Cleanup action that keeps only listed categories in a scene."""

    rule: Literal["include"] = Field("include", repr=False, init=False)
    categories: Categories
    """Categories retained in the scene during cleanup."""


CleanupSpec = Annotated[
    ExcludeCategoriesSpec | IncludeCategoriesSpec | PruneByRuleSpec, Field(discriminator="rule")
]
"""Discriminated union of all declarative cleanup actions applied before screening.

Cleanup specs can drop categories or prune agents based on nested screening
rules before the main scene and agent checks run.
"""


class ScreeningConfig(ResolvedConfig):
    """Declarative screening configuration composed from named rule maps."""

    cleanup: dict[str, CleanupSpec] = Field(default_factory=dict)
    """Named cleanup actions applied before screening checks are evaluated."""
    scene: dict[str, SceneCheckSpec] = Field(default_factory=dict)
    """Named scene-level screening rules."""
    agent: dict[str, AgentCheckSpec] = Field(default_factory=dict)
    """Named agent-level screening rules."""


class PartialScreeningConfig(ConfigPatch[ScreeningConfig]):
    """Patch model for replacing or extending named screening rule sets."""

    mode: Literal["replace", "extend"] | None = None
    """How this partial screening config should combine with an existing target."""
    remove: tuple[str, ...] | None = None
    """Named cleanup, scene, or agent rules to remove after merging."""
    cleanup: dict[str, CleanupSpec] | None = None
    """Cleanup rule overrides keyed by rule name."""
    scene: dict[str, SceneCheckSpec] | None = None
    """Scene-level rule overrides keyed by rule name."""
    agent: dict[str, AgentCheckSpec] | None = None
    """Agent-level rule overrides keyed by rule name."""
    full_config_type: type[ScreeningConfig] = Field(ScreeningConfig, repr=False, init=False)

    @override
    def merge_into(
        self, target: ScreeningConfig | None, *, exclude_none: bool = True
    ) -> ScreeningConfig:
        mode = self.mode or "extend"
        cleanup = target.cleanup if target is not None else {}
        scene = target.scene if target is not None else {}
        agent = target.agent if target is not None else {}
        if mode == "replace":
            cleanup = self.cleanup if self.cleanup is not None else {}
            scene = self.scene if self.scene is not None else {}
            agent = self.agent if self.agent is not None else {}
        elif mode == "extend":
            cleanup = {**cleanup, **(self.cleanup or {})}
            scene = {**scene, **(self.scene or {})}
            agent = {**agent, **(self.agent or {})}
        if self.remove:
            cleanup = {k: v for k, v in cleanup.items() if k not in self.remove}
            scene = {k: v for k, v in scene.items() if k not in self.remove}
            agent = {k: v for k, v in agent.items() if k not in self.remove}
        return ScreeningConfig(cleanup=cleanup, scene=scene, agent=agent)


class DatasetConfig(ResolvedConfig):
    """Full dataset/profile-style configuration schema."""

    scenes: ScenesConfig
    """Scene construction and temporal sampling settings."""
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    """Runtime execution settings such as worker count and chunk size."""
    screening: ScreeningConfig | None = Field(default=None)
    """Optional screening rules applied before scenes are emitted."""
    output: OutputConfig = Field(default_factory=OutputConfig)
    """Output encoding, schema, and backend-specific writer settings."""
    map: MapConfig = Field(default_factory=MapConfig)
    """Map extraction and interpolation settings for generated scenes."""
    read: ReadConfig = Field(default_factory=lambda: ReadConfig(ReadAll()))
    """Input selection configuration used to choose raw dataset sources."""
    assign: AssignConfig = Field(default_factory=lambda: AssignConfig(NoAssign()))
    """Output assignment configuration used to label generated scenes."""
    loader_options: dict[str, Any] | None = Field(default=None)
    """Dataset-specific loader options forwarded to the selected dataset plugin."""


class PartialDatasetConfigBase(ConfigBase):
    """Common optional fields shared by partial dataset-style configs.

    This base is reused by authored dataset entries and profile fragments.
    """

    scenes: PartialScenesConfig | None = Field(default=None)
    """Partial scene construction overrides to merge into the target config."""
    runtime: PartialRuntimeConfig | None = Field(default=None)
    """Partial runtime execution overrides to merge into the target config."""
    screening: PartialScreeningConfig | Literal[False] | None = Field(default=None)
    """Partial screening rule overrides to merge into the target config."""
    output: PartialOutputConfig | None = Field(default=None)
    """Partial output writer overrides to merge into the target config."""
    map: PartialMapConfig | None = Field(default=None)
    """Partial map extraction overrides to merge into the target config."""
    read: ReadConfig | None = Field(default=None)
    """Replacement input selection strategy for the target dataset config."""
    assign: AssignConfig | None = Field(default=None)
    """Replacement output assignment strategy for the target dataset config."""
    loader_options: dict[str, Any] | None = Field(default=None)
    """Dataset-specific loader option overrides for the target config."""


class PartialDatasetConfig(PartialDatasetConfigBase, ConfigPatch[DatasetConfig]):
    """Patch model for applying partial values to a full dataset config.

    The merge strategy preserves existing nested defaults unless a matching
    partial model is provided.
    """

    full_config_type: type[DatasetConfig] = DatasetConfig

    @override
    def merge_into(
        self, target: DatasetConfig | None, *, exclude_none: bool = True
    ) -> DatasetConfig:
        if target is None:
            msg = "Defaults must be provided to apply a PartialDatasetConfig."
            raise ValueError(msg)

        return DatasetConfig(
            scenes=self.scenes.merge_into(target.scenes) if self.scenes else target.scenes,
            runtime=self.runtime.merge_into(target.runtime) if self.runtime else target.runtime,
            screening=apply_optional(self.screening, target.screening),
            loader_options=(
                self.loader_options if self.loader_options is not None else target.loader_options
            ),
            output=self.output.merge_into(target.output) if self.output else target.output,
            map=self.map.merge_into(target.map) if self.map else target.map,
            read=self.read if self.read is not None else target.read,
            assign=self.assign if self.assign is not None else target.assign,
        )


class RuntimeOverride(ConfigPatch[PartialDatasetConfig]):
    """User-supplied overrides layered on top of a dataset's base config.

    `RuntimeOverride` is the bridge between unstructured runtime inputs such as
    CLI flags and the strongly typed configuration models used by the execution
    pipeline. It only carries override fields that are safe to merge into an
    existing dataset config at run time.
    """

    runtime: PartialRuntimeConfig | None = None
    read: ReadConfig | None = None
    assign: AssignConfig | None = None
    output: PartialOutputConfig | None = None
    full_config_type: type[PartialDatasetConfig] = Field(
        default=PartialDatasetConfig, init=False, repr=False
    )

    @staticmethod
    def _validate_read_inputs(
        *, read_strategy: ReadStrategy | None, read_split: list[DatasetSplit] | None
    ) -> None:
        if read_split is None:
            return
        if read_strategy != "native":
            msg = "`read_split` requires `read_strategy='native'`."
            raise ConfigurationError(msg)

    @staticmethod
    def _validate_assign_inputs(
        *,
        assign_strategy: AssignStrategy | None,
        ratio: tuple[float, float, float] | None,
        gap: int | None,
        segments: int | None,
    ) -> None:
        if assign_strategy is None and any(value is not None for value in (ratio, gap, segments)):
            msg = "Assignment options require `assign_strategy` to be set."
            raise ConfigurationError(msg)

        weighted = {"scene", "source", "time", "shuffled-time"}
        time_based = {"time", "shuffled-time"}

        if ratio is not None and assign_strategy not in weighted:
            msg = "`ratio` is only valid for scene, source, time, and shuffled-time assignment."
            raise ConfigurationError(msg)
        if gap is not None and assign_strategy not in time_based:
            msg = "`gap` is only valid for time and shuffled-time assignment."
            raise ConfigurationError(msg)
        if segments is not None and assign_strategy != "shuffled-time":
            msg = "`segments` is only valid for shuffled-time assignment."
            raise ConfigurationError(msg)
        if assign_strategy in weighted and ratio is None:
            msg = f"`ratio` is required when `assign_strategy='{assign_strategy}'`."
            raise ConfigurationError(msg)
        if assign_strategy == "shuffled-time" and segments is None:
            msg = "`segments` is required when `assign_strategy='shuffled-time'`."
            raise ConfigurationError(msg)

    @classmethod
    def from_inputs(
        cls,
        read_strategy: ReadStrategy | None = None,
        read_split: list[DatasetSplit] | None = None,
        assign_strategy: AssignStrategy | None = None,
        jobs: int | Literal["auto"] | None = None,
        trajectory_schema: str | None = None,
        ratio: tuple[float, float, float] | None = None,
        gap: int | None = None,
        segments: int | None = None,
    ) -> RuntimeOverride:
        """Construct a runtime override from raw inputs.

        Inputs are based on raw CLI arguments that are not validated.

        !!! note "`PartialDatasetConfig` conversion"
            The `merge_into` method can be used to convert this `RuntimeOverride`
            into a `PartialDatasetConfig` that can be merged with the dataset's
            default config. This allows for applying overrides without needing
            to specify the full config structure.

        Parameters
        ----------
        read_strategy: ReadStrategy or None
            The strategy to use for selecting raw dataset inputs. If None, no
            override is applied.
        read_split: list of DatasetSplit or None
            The dataset-native partitions to read from the dataset. Only
            applicable if read_strategy is "native". If None, no override is
            applied.
        assign_strategy: AssignStrategy or None
            The strategy to use for assigning output split labels. If None, no
            override is applied.
        jobs: int or Literal["auto"] or None
            The number of parallel jobs to use for loading and processing the
            dataset. If None, no override is applied. If set to "auto", the
            number of jobs will be set to the number of CPU cores.
        trajectory_schema: str or None
            The trajectory schema to use for the output trajectories. If None,
            no override is applied.
        ratio: tuple of three floats or None
            The ratio of train, validation, and test assignments to use.
            Only applicable if assign_strategy is "scene", "source", "time",
            or "shuffled-time". If None, no override is applied.
        gap: int or None
            The gap (in frames) to use between partitions when using "time" or
            "shuffled-time" assignment strategies. If None, no override is applied.
        segments: int or None
            The number of segments to use for shuffled-time assignment. If None,
            no override is applied.

        Returns
        -------
        RuntimeOverride
            A runtime override object containing the specified overrides.

        """
        cls._validate_read_inputs(read_strategy=read_strategy, read_split=read_split)
        cls._validate_assign_inputs(
            assign_strategy=assign_strategy, ratio=ratio, gap=gap, segments=segments
        )

        read_data = {"strategy": read_strategy, "splits": read_split}
        read_data = {k: v for k, v in read_data.items() if v is not None}
        read_config = ReadConfig.model_validate(read_data) if "strategy" in read_data else None

        assign_data = {
            "strategy": assign_strategy,
            "ratio": {"train": ratio[0], "val": ratio[1], "test": ratio[2]} if ratio else None,
            "gap": gap,
            "segments": segments,
        }
        assign_data = {k: v for k, v in assign_data.items() if v is not None}
        assign_config = (
            AssignConfig.model_validate(assign_data) if "strategy" in assign_data else None
        )

        return cls(
            runtime=PartialRuntimeConfig(jobs=jobs) if jobs is not None else None,
            read=read_config,
            assign=assign_config,
            output=PartialOutputConfig(trajectory_schema=trajectory_schema)
            if trajectory_schema is not None
            else None,
        )
