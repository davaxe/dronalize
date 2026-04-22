"""Public TOML file schema."""

from __future__ import annotations

import contextlib

from pydantic import Field

from dronalize.config.base import ConfigBase
from dronalize.config.models import DatasetConfig, PartialDatasetConfig, PartialDatasetConfigBase
from dronalize.core.errors import ConfigurationError


class PartialDatasetEntryConfig(PartialDatasetConfigBase):
    """Dataset-local authored config."""

    uses: tuple[str, ...] | None = None

    def partial_config(self) -> PartialDatasetConfig:
        """Return config without the uses field."""
        return PartialDatasetConfig(
            scenes=self.scenes,
            runtime=self.runtime,
            screening=self.screening,
            dataset=self.dataset,
            output=self.output,
            map=self.map,
            read=self.read,
            assign=self.assign,
        )


class ProcessingConfig(ConfigBase):
    """Root config model for processing requests.

    This model is designed to be loaded from a TOML file with potentially
    incomplete dataset entries, which can be resolved to complete
    [`DatasetConfig`][dronalize.config.models.DatasetConfig]s using the
    `resolve` method.

    To load a `ProcessingConfig` from a TOML file, use the
    [`parse_config`][dronalize.config.reader.parse_config] function.

    """

    profiles: dict[str, PartialDatasetConfig] = Field(default_factory=dict)
    datasets: dict[str, PartialDatasetEntryConfig] = Field(default_factory=dict)

    def resolve(self, dataset: str, dataset_config: DatasetConfig) -> DatasetConfig:
        """Resolve a specific dataset.

        Parameters
        ----------
        dataset : str
            The name of the dataset to resolve, e.g. `"argoverse1"` or `"vod"`.
        dataset_config : DatasetConfig
            The full configuration before resolution.
        """
        partial = self.datasets.get(dataset) if self.datasets else None
        if partial is None:
            return dataset_config

        for use in partial.uses or ():
            profile = self.profiles.get(use)
            if profile is None:
                msg = f"Profile '{use}' not found for dataset '{dataset}'"
                raise ConfigurationError(msg)
            dataset_config = profile.apply_to(dataset_config)

        return partial.partial_config().apply_to(dataset_config)

    def extract(self, dataset: str) -> DatasetConfig | None:
        """Extract the dataset config for a specific dataset without resolution.

        There are two failure modes for this method that both result in `None`
        being returned:

        1. the dataset is not found in the file, and
        2. the dataset is found but resolution fails due to incomplete config
           values.

        Parameters
        ----------
        dataset : str
            The name of the dataset to extract, e.g. `"argoverse1"` or `"vod"`.

        Returns
        -------
        DatasetConfig | None
            The dataset config for the specified dataset, or `None` if not found
            or if resolution fails.
        """
        partial = self.datasets.get(dataset) if self.datasets else None
        if partial is None:
            return None
        with contextlib.suppress(ValueError):
            return partial.partial_config().apply_to(None)
        return None
