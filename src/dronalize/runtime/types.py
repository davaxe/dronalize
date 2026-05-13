"""Public runtime data models shared by API and internal execution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dronalize.config.models import MapConfig, effective_scene_window
from dronalize.config.runtime import RuntimeOverride
from dronalize.core.errors import ConfigurationError
from dronalize.core.scene.model import derived_trajectory_fields
from dronalize.core.scene.schema import TrajectorySchema, get_trajectory_schema
from dronalize.io.formats import StorageBackend, storage_backend_name
from dronalize.io.manifest import DatasetManifest, package_version, write_manifest
from dronalize.processing.models import AssignmentRequest, LoaderRequest, ReadRequest

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.config.models import DatasetConfig, MDSOutputConfig, OutputConfig, RuntimeConfig
    from dronalize.datasets.registry import DatasetSpec
    from dronalize.processing.loading.models import DatasetOptionsModel


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Final result of a processing run.

    `processed_sources` counts how many source units the executor started
    processing, even when a scene limit stops the run before a source is fully
    exhausted.
    """

    dataset: str
    """Dataset key used for the run."""
    output_dir: Path
    """Root directory where processed output was written."""
    storage_backend: StorageBackend | str
    """Storage backend used for the exported records."""
    processed_sources: int
    """Number of raw source units the executor started processing."""
    candidate_scenes: int
    """Number of scene candidates materialized before screening."""
    selected_scenes: int
    """Number of scenes accepted for output after screening and limits."""
    split_counts: dict[str, int]
    """Accepted scene count per output split."""


@dataclass(frozen=True)
class OutputPlan:
    """Plan for output configuration."""

    inner: OutputConfig
    """Resolved output configuration used by the writer and manifest."""

    def precision(self) -> type[np.float32 | np.float64]:
        """Return the floating point precision for this output plan."""
        if self.inner.precision == "float32":
            return np.float32
        return np.float64

    @property
    def mds(self) -> MDSOutputConfig:
        """Return the MDS output config for this output plan."""
        return self.inner.mds

    @property
    def recenter_positions(self) -> bool:
        """Return whether this output plan requests recentering of agent positions."""
        return self.inner.recenter_positions

    @cached_property
    def trajectory_schema(self) -> TrajectorySchema:
        """Return the trajectory schema for this output plan."""
        return get_trajectory_schema(self.inner.trajectory_schema)


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

    descriptor: DatasetSpec
    """Resolved dataset specification."""
    data_root: Path
    """Input dataset root."""
    output_dir: Path
    """Output dataset root."""
    storage_backend: StorageBackend | str
    """Storage backend selected for writing output records."""
    resolved_config: DatasetConfig
    """Dataset config after defaults, config files, and overrides are merged."""
    runtime: RuntimeConfig
    """Resolved runtime execution settings."""
    output: OutputPlan
    """Resolved output settings and derived output schema."""
    loader: LoaderRequest
    """Loader-facing subset of the resolved configuration."""
    assignment: AssignmentRequest
    """Compiled split-assignment request."""
    map: MapConfig | None
    """Resolved map configuration, or `None` when map output is disabled."""
    effective_history_frames: int
    """History frame count after resampling/window configuration is applied."""
    effective_future_frames: int
    """Future frame count after resampling/window configuration is applied."""
    effective_sample_time: float
    """Sample interval in seconds after resampling is applied."""
    limit: int | None = None
    """Optional maximum number of selected scenes to write."""
    seed: int | None = None
    """Optional seed used by deterministic runtime choices."""

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
        return DatasetManifest(
            dataset=self.dataset,
            storage_backend=storage_backend_name(self.storage_backend),
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
            future_frames=self.effective_future_frames,
            history_frames=self.effective_history_frames,
            has_map=self.map is not None,
            derived_features=tuple(
                field.to_str()
                for field in derived_trajectory_fields(
                    self.descriptor.native_schema,
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


def compile_loader_request(
    *, descriptor: DatasetSpec, resolved_config: DatasetConfig, include_map: bool | None
) -> LoaderRequest:
    """Compile the loader-facing request for one resolved dataset config."""
    loader_options: DatasetOptionsModel = descriptor.parse_loader_options(
        resolved_config.loader_options
    )
    map_config = (
        None
        if (include_map is False or not descriptor.feature_support.map)
        else resolved_config.map
    )
    return LoaderRequest(
        scenes=resolved_config.scenes,
        screening=resolved_config.screening,
        read=ReadRequest.from_config(
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

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

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
    """Optional maximum number of selected scenes to write."""
    seed: int | None = None
    """Optional seed used by deterministic runtime choices."""
    input_dir_exists: bool = True
    """Whether request resolution should require `input_dir` to exist."""


def compile_effective_scene_metrics(config: DatasetConfig) -> tuple[int, int, float]:
    """Return the effective scene window and sample time for one resolved config."""
    return effective_scene_window(config.scenes)
