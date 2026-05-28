from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from pydantic import Field

from dronalize.config.base import ConfigBase
from dronalize.config.models import DatasetConfig, PartialDatasetConfig, PartialDatasetConfigBase
from dronalize.core.errors import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib  # pyright: ignore[reportUnreachable]
else:
    import tomli as tomllib


def parse_config(path: Path) -> ProjectConfig:
    """Parse configuration from a TOML file.

    !!! note "Completeness of the returned config"
        By design, this loader returns a validated but potentially incomplete configuration for
        specific datasets.

        See the
        [`ProjectConfig.resolve_dataset_config`][dronalize.config.ProjectConfig.resolve_dataset_config]
        method for applying dataset-specific overrides returned by this loader
        on top of a fully resolved `DatasetConfig` to get a complete
        configuration for a specific dataset.

    Parameters
    ----------
    path : Path
        Path to the TOML configuration file.

    Returns
    -------
    ProjectConfig
        The parsed configuration, validated and ready for dataset-specific resolution.
    """
    with path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            msg = f"Invalid TOML in config file '{path}': {exc}"
            raise ConfigurationError(msg) from exc
    return ProjectConfig.model_validate(data)


class DatasetConfigEntry(PartialDatasetConfigBase):
    """Dataset-local authored config."""

    uses: tuple[str, ...] | None = None

    def to_partial_dataset_config(self) -> PartialDatasetConfig:
        """Return config without the uses field."""
        return PartialDatasetConfig(
            scenes=self.scenes,
            runtime=self.runtime,
            screening=self.screening,
            loader_options=self.loader_options,
            output=self.output,
            map=self.map,
            read=self.read,
            assign=self.assign,
        )


class ProjectConfig(ConfigBase):
    """Root config model for processing requests.

    This model is designed to be loaded from a TOML file with potentially
    incomplete dataset entries, which can be resolved to complete
    [`DatasetConfig`][dronalize.config.models.DatasetConfig]s using the
    `resolve` method.

    To load a `ProjectConfig` from a TOML file, use the
    [`parse_config`][dronalize.config.reader.parse_config] function.

    """

    defaults: DatasetConfigEntry | None = Field(default=None)
    profiles: dict[str, PartialDatasetConfig] = Field(default_factory=dict)
    datasets: dict[str, DatasetConfigEntry] = Field(default_factory=dict)

    def resolve_dataset_config(self, dataset: str, dataset_config: DatasetConfig) -> DatasetConfig:
        """Resolve a specific dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset to resolve, e.g. `"argoverse1"` or `"vod"`.
        dataset_config : DatasetConfig
            The full configuration before resolution.
        """
        dataset_config = self._apply_entry(
            entry=self.defaults, target=dataset_config, context="defaults"
        )
        return self._apply_entry(
            entry=self.datasets.get(dataset) if self.datasets else None,
            target=dataset_config,
            context=f"dataset '{dataset}'",
        )

    def _apply_entry(
        self, *, entry: DatasetConfigEntry | None, target: DatasetConfig, context: str
    ) -> DatasetConfig:
        if entry is None:
            return target

        for use in entry.uses or ():
            profile = self.profiles.get(use)
            if profile is None:
                msg = f"Profile '{use}' not found for {context}"
                raise ConfigurationError(msg)
            target = profile.merge_into(target)

        return entry.to_partial_dataset_config().merge_into(target)
