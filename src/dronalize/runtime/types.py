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
from dronalize.io.formats import StorageBackend
from dronalize.io.manifest import DatasetManifest, write_manifest
from dronalize.processing.models import LoaderRequest, SplitRequest

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.config.models import (
        DatasetConfig,
        MDSOutputConfig,
        OutputConfig,
        RuntimeConfig,
    )
    from dronalize.datasets.registry import DatasetSpec


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Final result of a processing run."""

    dataset: str
    output_dir: Path
    storage_backend: StorageBackend
    processed_sources: int
    processed_scenes: int
    split_counts: dict[str, int]


@dataclass(frozen=True)
class OutputPlan:
    """Plan for output configuration."""

    inner: OutputConfig

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
    """Initial compiled runtime plan.

    !!! info "Do not construct this directly"
        This is not meant to be directly constructed. Instead, it should be
        built using the
        [`resolve_request`][dronalize.runtime.api.resolve_request] function,
        which takes [`ExecutionRequest`][dronalize.runtime.ExecutionRequest] and
        produces a fully resolved plan ready for execution.

    """

    descriptor: DatasetSpec
    data_root: Path
    output_dir: Path
    storage_backend: StorageBackend
    resolved_config: DatasetConfig
    runtime: RuntimeConfig
    output: OutputPlan
    loader: LoaderRequest
    map: MapConfig | None
    effective_history_frames: int
    effective_future_frames: int
    effective_sample_time: float
    limit: int | None = None
    seed: int | None = None

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
            precision=export_config.precision,
            feature_columns=self.output.trajectory_schema.feature_columns(),
            trajectory_schema=self.output.trajectory_schema.name,
            recenter_positions=export_config.recenter_positions,
            source_trajectory_schema=self.descriptor.native_schema.name,
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
    *,
    descriptor: DatasetSpec,
    resolved_config: DatasetConfig,
    seed: int | None,
    include_map: bool | None,
) -> LoaderRequest:
    """Compile the loader-facing request for one resolved dataset config."""
    dataset_options = descriptor.parse_dataset_config(resolved_config.dataset)
    map_config = None if (include_map is False or not descriptor.has_map) else resolved_config.map
    return LoaderRequest(
        scenes=resolved_config.scenes,
        screening=resolved_config.screening,
        split=SplitRequest.from_config(resolved_config.split, seed=seed),
        dataset=dataset_options,
        map=map_config,
        native_splits=descriptor.native_splits or None,
    )


class ExecutionRequest(BaseModel):
    """Normalized user request for one dataset processing job."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    input_dir: Path
    output_dir: Path
    storage_backend: StorageBackend | str = StorageBackend.PICKLE
    config_path: Path | None = None
    overrides: RuntimeOverride = Field(default_factory=RuntimeOverride)
    include_map: bool | None = None
    limit: int | None = None
    seed: int | None = None
    input_dir_exists: bool = True


def compile_effective_scene_metrics(config: DatasetConfig) -> tuple[int, int, float]:
    """Return the effective scene window and sample time for one resolved config."""
    return effective_scene_window(config.scenes)
