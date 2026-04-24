"""Full dataset configuration models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from typing_extensions import override

from dronalize.config.base import ConfigBase, FullConfig, PartialConfig, apply_optional
from dronalize.config.models.map import MapConfig, PartialMapConfig
from dronalize.config.models.output import OutputConfig, PartialOutputConfig
from dronalize.config.models.runtime import PartialRuntimeConfig, RuntimeConfig
from dronalize.config.models.scenes import PartialScenesConfig, ScenesConfig  # noqa: TC001
from dronalize.config.models.screening import PartialScreeningConfig, ScreeningConfig  # noqa: TC001
from dronalize.config.models.split import AssignConfig, NoAssign, ReadAll, ReadConfig


class DatasetConfig(FullConfig):
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
    dataset: dict[str, Any] | None = Field(default=None)
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
    dataset: dict[str, Any] | None = Field(default=None)
    """Dataset-specific loader option overrides for the target config."""


class PartialDatasetConfig(PartialDatasetConfigBase, PartialConfig[DatasetConfig]):
    """Patch model for applying partial values to a full dataset config.

    The merge strategy preserves existing nested defaults unless a matching
    partial model is provided.
    """

    full_config_type: type[DatasetConfig] = DatasetConfig

    @override
    def apply_to(self, target: DatasetConfig | None, *, exclude_none: bool = True) -> DatasetConfig:
        if target is None:
            msg = "Defaults must be provided to apply a PartialDatasetConfig."
            raise ValueError(msg)

        return DatasetConfig(
            scenes=self.scenes.apply_to(target.scenes) if self.scenes else target.scenes,
            runtime=self.runtime.apply_to(target.runtime) if self.runtime else target.runtime,
            screening=apply_optional(self.screening, target.screening),
            dataset=self.dataset if self.dataset is not None else target.dataset,
            output=self.output.apply_to(target.output) if self.output else target.output,
            map=self.map.apply_to(target.map) if self.map else target.map,
            read=self.read if self.read is not None else target.read,
            assign=self.assign if self.assign is not None else target.assign,
        )
