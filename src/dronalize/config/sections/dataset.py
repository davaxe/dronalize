from __future__ import annotations

from typing import Any

from pydantic import Field
from typing_extensions import override

from dronalize.config.base import ConfigBase, FullConfig, PartialConfig
from dronalize.config.sections.map import MapConfig, PartialMapConfig
from dronalize.config.sections.output import OutputConfig, PartialOutputConfig
from dronalize.config.sections.runtime import PartialRuntimeConfig, RuntimeConfig
from dronalize.config.sections.scenes import PartialScenesConfig, ScenesConfig  # noqa: TC001
from dronalize.config.sections.screening import (  # noqa: TC001
    PartialScreeningConfig,
    ScreeningConfig,
)
from dronalize.config.sections.split import NoSplitConfig, SplitConfig


class DatasetConfig(FullConfig):
    """Full dataset/profile-style configuration schema."""

    scenes: ScenesConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    screening: ScreeningConfig | None = Field(default=None)
    output: OutputConfig = Field(default_factory=OutputConfig)
    map: MapConfig = Field(default_factory=MapConfig)
    split: SplitConfig = Field(default_factory=lambda: SplitConfig(NoSplitConfig()))
    dataset: dict[str, Any] | None = Field(default=None)


class PartialDatasetConfigBase(ConfigBase):
    scenes: PartialScenesConfig | None = None
    runtime: PartialRuntimeConfig | None = None
    screening: PartialScreeningConfig | None = None
    output: PartialOutputConfig | None = None
    map: PartialMapConfig | None = None
    split: SplitConfig | None = None
    dataset: dict[str, Any] | None = None


class PartialDatasetConfig(PartialDatasetConfigBase, PartialConfig[DatasetConfig]):
    """Partial dataset/profile-style configuration schema."""

    full_config_type: type[DatasetConfig] = DatasetConfig

    @override
    def apply_to(self, target: DatasetConfig | None, *, exclude_none: bool = True) -> DatasetConfig:
        if target is None:
            msg = "Defaults must be provided to apply a PartialDatasetConfig."
            raise ValueError(msg)

        return DatasetConfig(
            scenes=self.scenes.apply_to(target.scenes) if self.scenes else target.scenes,
            runtime=self.runtime.apply_to(target.runtime) if self.runtime else target.runtime,
            screening=(
                self.screening.apply_to(target.screening) if self.screening else target.screening
            ),
            dataset=self.dataset if self.dataset is not None else target.dataset,
            output=self.output.apply_to(target.output) if self.output else target.output,
            map=self.map.apply_to(target.map) if self.map else target.map,
            split=self.split if self.split is not None else target.split,
        )
