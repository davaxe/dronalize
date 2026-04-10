"""Public TOML file schema."""

from __future__ import annotations

import contextlib

from pydantic import Field

from dronalize.config.base import ConfigBase
from dronalize.config.sections import DatasetConfig, PartialDatasetConfig, PartialDatasetConfigBase

# --------------------------------------------------------------------------------------
# Dataset entry
# --------------------------------------------------------------------------------------


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
            split=self.split,
        )


# --------------------------------------------------------------------------------------
# Root file
# --------------------------------------------------------------------------------------


class ProjectConfig(ConfigBase):
    """Partial root public TOML schema."""

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
                raise ValueError(msg)
            dataset_config = profile.apply_to(dataset_config)

        return partial.partial_config().apply_to(dataset_config)

    def extract(self, dataset: str) -> DatasetConfig | None:
        """Extract the dataset config for a specific dataset without resolution.

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
