"""Public runtime data models shared by API and internal execution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path  # noqa: TC003 - Pydantic resolves this forward reference at runtime.
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dronalize.config.models import RuntimeOverride, effective_scene_window
from dronalize.core.errors import ConfigurationError
from dronalize.core.scene.model import derived_trajectory_fields
from dronalize.core.scene.schema import POSITIONS_ONLY, TrajectorySchema, get_trajectory_schema
from dronalize.datasets.registry import dataset_id_for_name, dataset_names_by_id
from dronalize.io.base import (
    RecordTransform,
    SceneTransform,
    StorageBackend,
    validate_transform_choice,
)
from dronalize.io.manifest import DatasetManifest, package_version, write_manifest
from dronalize.processing.models import LoaderPlan, ReadSelection, SplitAssignmentPlan

if TYPE_CHECKING:
    from dronalize.config.models import (
        DatasetConfig,
        MapConfig,
        MDSOutputConfig,
        OutputConfig,
        RuntimeConfig,
    )
    from dronalize.datasets.registry import DatasetDescriptor
    from dronalize.processing.loading.models import LoaderOptionsModel
    from dronalize.runtime.state import ExecutionStats


@dataclass(frozen=True, slots=True)
class CleanupRemovalSummary:
    """Aggregate cleanup-removal statistics over candidate scenes with cleanup stats.

    Cleanup is recorded before screening rejection is applied, so ``scene_count``
    does not necessarily match the number of written scenes.
    """

    scene_count: int
    total_rows_removed: int
    average_rows_removed_per_scene: float
    min_rows_removed_per_scene: int
    max_rows_removed_per_scene: int
    total_agents_removed: int
    average_agents_removed_per_scene: float
    min_agents_removed_per_scene: int
    max_agents_removed_per_scene: int


@dataclass(frozen=True, slots=True)
class CleanupSummary:
    """Final cleanup statistics, optionally broken down by cleanup rule."""

    overall: CleanupRemovalSummary
    by_rule: dict[str, CleanupRemovalSummary]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Final result of one dataset execution.

    ``stats`` is the canonical source, scene, split, screening, and cleanup
    progress-counter payload. ``cleanup_summary`` contains the heavier final
    diagnostic cleanup summary.
    """

    dataset: str
    """Dataset that was executed."""
    output_dir: Path
    """Directory where output was written."""
    storage_backend: StorageBackend | str
    """Storage backend used for writing output."""
    stats: ExecutionStats
    """Final source, scene, split, screening, and cleanup counters."""
    scene_limit: int | None
    """Output scene limit used for this run, if any."""
    cleanup_summary: CleanupSummary | None
    """Final cleanup summary over candidate scenes with cleanup statistics."""
    elapsed_time_seconds: float
    """Wall-clock execution time in seconds."""


@dataclass(frozen=True)
class OutputPlan:
    """Plan for output configuration."""

    config: OutputConfig
    """Resolved output configuration used by the writer and manifest."""
    default_observation_length: int | None = None
    """Default split point to store on each output record, if known."""

    def precision(self) -> type[np.float32 | np.float64]:
        """Return the floating point precision for this output plan."""
        if self.config.precision == "float32":
            return np.float32
        return np.float64

    @property
    def mds(self) -> MDSOutputConfig:
        """Return the MDS output config for this output plan."""
        return self.config.mds

    @property
    def recenter_positions(self) -> bool:
        """Return whether this output plan requests recentering of agent positions."""
        return self.config.recenter_positions

    @cached_property
    def trajectory_schema(self) -> TrajectorySchema:
        """Return the trajectory schema for this output plan."""
        return get_trajectory_schema(self.config.trajectory_schema)


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Fully resolved runtime plan produced from an execution request.

    !!! info "Do not construct this directly"
        This is not meant to be directly constructed. Instead, it should be
        built using the
        [`resolve_request`][dronalize.runtime.api.resolve_request] function,
        which takes [`ExecutionRequest`][dronalize.runtime.ExecutionRequest] and
        produces a fully resolved plan ready for execution.

    """

    descriptor: DatasetDescriptor
    """Resolved dataset descriptor."""
    data_root: Path
    """Input dataset root."""
    output_dir: Path
    """Output dataset root."""
    storage_backend: StorageBackend
    """Storage backend selected for writing output records."""
    resolved_config: DatasetConfig
    """Dataset config after defaults, config files, and overrides are merged."""
    runtime: RuntimeConfig
    """Resolved runtime execution settings."""
    output: OutputPlan
    """Resolved output settings and derived output schema."""
    loader: LoaderPlan
    """Loader-facing subset of the resolved configuration."""
    assignment: SplitAssignmentPlan
    """Compiled split-assignment request."""
    map: MapConfig | None
    """Resolved map configuration, or `None` when map output is disabled."""
    effective_horizon_frames: int
    """Horizon frame count after resampling/window configuration is applied."""
    effective_default_observation_length: int | None
    """Default reader/adaptor split point after resampling, if configured."""
    effective_sample_time: float
    """Effective `sample_time` interval in seconds after resampling is applied."""
    output_transform: OutputTransform[object] | None = None
    """Optional custom persisted-output configuration for writer backends."""
    limit: int | None = None
    """Optional maximum number of scenes to write."""
    seed: int | None = None
    """Optional seed used by deterministic runtime choices."""
    overwrite: bool = False
    """Whether execution may replace an existing non-empty output directory."""

    def __post_init__(self) -> None:
        """Validate the runtime plan after initialization."""
        if self.limit == 0:
            object.__setattr__(self, "limit", None)
        if self.limit is not None and self.limit <= 0:
            msg = f"Limit must be a positive integer, got {self.limit}."
            raise ConfigurationError(msg)

    @property
    def dataset(self) -> str:
        """Return the dataset key for this plan."""
        return self.descriptor.name

    @property
    def parallel(self) -> bool:
        """Return whether the runtime plan requests parallel execution."""
        return self.runtime.jobs > 1

    @property
    def workers(self) -> int:
        """Return the number of workers requested by the runtime plan."""
        return self.runtime.jobs

    def manifest(self) -> DatasetManifest:
        """Return the dataset manifest for this plan."""
        export_config: OutputConfig = self.resolved_config.output
        derivation_source = trajectory_schema_after_transforms(
            self.descriptor.native_schema, self.resolved_config
        )
        return DatasetManifest(
            dataset=self.dataset,
            dataset_names=(
                dataset_names_by_id()
                if dataset_id_for_name(self.dataset) is not None
                else (self.dataset,)
            ),
            storage_backend=self.storage_backend.value,
            dronalize_version=package_version(),
            precision=export_config.precision,
            feature_columns=self.output.trajectory_schema.feature_columns(),
            trajectory_schema=self.output.trajectory_schema.name,
            trajectory_schema_fields=self.output.trajectory_schema.semantic_fields(),
            recenter_positions=export_config.recenter_positions,
            source_trajectory_schema=self.descriptor.native_schema.name,
            source_trajectory_schema_fields=self.descriptor.native_schema.semantic_fields(),
            sample_time=self.effective_sample_time,
            original_sample_time=self.resolved_config.scenes.sample_time,
            horizon_frames=self.effective_horizon_frames,
            default_observation_length=self.effective_default_observation_length,
            has_map=self.map is not None,
            derived_features=tuple(
                field.to_str()
                for field in derived_trajectory_fields(
                    derivation_source,
                    self.output.trajectory_schema,
                    sample_time=self.resolved_config.scenes.sample_time,
                )
            ),
        )

    def manifest_roots(self) -> tuple[Path, ...]:
        """Return directories that should receive the generated manifest."""
        return (self.output_dir,)

    def write_manifests(self) -> None:
        """Persist the compiled manifest to all configured targets."""
        manifest = self.manifest()
        for root in self.manifest_roots():
            write_manifest(root, manifest)


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True)
class OutputTransform(Generic[PayloadT]):
    """Python API configuration for custom persisted output payloads.

    `record_transform` is the preferred hook because it receives the canonical
    `SceneRecord` after Dronalize has applied normal output semantics.
    `scene_transform` is an expert escape hatch for deriving payloads directly
    from runtime `Scene` objects.
    """

    record_transform: RecordTransform[PayloadT] | None = None
    """Optional transform from canonical `SceneRecord` to persisted payload."""
    scene_transform: SceneTransform[PayloadT] | None = None
    """Optional transform from runtime `Scene` to persisted payload."""
    mds_columns: dict[str, str] | None = None
    """MDS column schema required when custom rows are written to MDS."""

    def __post_init__(self) -> None:
        """Validate that only one transform mode is configured."""
        validate_transform_choice(
            record_transform=self.record_transform, scene_transform=self.scene_transform
        )


def build_loader_plan(
    *, descriptor: DatasetDescriptor, resolved_config: DatasetConfig, include_map: bool | None
) -> LoaderPlan:
    """Compile the loader-facing request for one resolved dataset config."""
    loader_options: LoaderOptionsModel = descriptor.parse_loader_options(
        resolved_config.loader_options
    )
    map_config = (
        None
        if (include_map is False or not descriptor.feature_support.map)
        else resolved_config.map
    )
    return LoaderPlan(
        scenes=resolved_config.scenes,
        screening=resolved_config.screening,
        read=ReadSelection.from_config(
            resolved_config.read, supported_native_splits=descriptor.supported_native_splits
        ),
        loader_options=loader_options,
        map=map_config,
    )


class ExecutionRequest(BaseModel):
    """User-facing request for one dataset processing run.

    The request is intentionally small: it names the dataset, input/output
    paths, optional config file, and runtime overrides. Pass it to
    [`resolve_request`][dronalize.runtime.api.resolve_request] to obtain an
    [`ExecutionPlan`][dronalize.runtime.ExecutionPlan], or to
    [`execute_request`][dronalize.runtime.api.execute_request] to run directly.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    dataset: str
    """Dataset registry key, such as `a43` or `waymo`."""
    input_dir: Path
    """Root directory containing the raw dataset files."""
    output_dir: Path
    """Directory where processed output should be written."""
    storage_backend: StorageBackend | str = StorageBackend.PICKLE
    """Output storage backend. Built-in values are `pickle` and `mds`."""
    config_path: Path | None = None
    """Optional TOML config file applied on top of the dataset defaults."""
    overrides: RuntimeOverride = Field(default_factory=RuntimeOverride)
    """Programmatic runtime overrides applied after the config file."""
    include_map: bool | None = None
    """Override for map output. `None` uses the resolved dataset config."""
    limit: int | None = None
    """Optional maximum number of scenes to write."""
    seed: int | None = None
    """Optional seed used by deterministic runtime choices."""
    overwrite: bool = False
    """Whether execution may replace an existing non-empty output directory."""
    input_dir_exists: bool = True
    """Whether request resolution should require `input_dir` to exist."""
    output_transform: OutputTransform[object] | None = None
    """Customized output transform configuration."""


def resolve_effective_scene_window(config: DatasetConfig) -> tuple[int, int | None, float]:
    """Return the effective scene window and `sample_time` for one resolved config."""
    return effective_scene_window(config.scenes)


def trajectory_schema_after_transforms(
    native_schema: TrajectorySchema, config: DatasetConfig
) -> TrajectorySchema:
    """Return fields that remain semantically valid after temporal transforms."""
    resample = config.scenes.resample
    if resample is None:
        return native_schema
    if (
        resample.up == 1
        and resample.down == 1
        and not resample.emit_velocity
        and not resample.emit_acceleration
    ):
        return native_schema
    return POSITIONS_ONLY
