from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from dronalize.config.base import PartialConfig
from dronalize.config.models import (
    PartialDatasetConfig,
    PartialOutputConfig,
    PartialRuntimeConfig,
    SplitConfig,
)

if TYPE_CHECKING:
    from dronalize.core.categories import DatasetSplit

SplitStrategy = Literal["none", "native", "scene", "source", "time", "shuffled-time"]
"""Supported runtime split-strategy override names accepted by the CLI layer.

The string values map directly to the concrete split config models in
[`dronalize.config.models.split`][].
"""


class RuntimeOverride(PartialConfig[PartialDatasetConfig]):
    """User-supplied overrides layered on top of a dataset's base config.

    `RuntimeOverride` is the bridge between unstructured runtime inputs such as
    CLI flags and the strongly typed configuration models used by the execution
    pipeline. It only carries override fields that are safe to merge into an
    existing dataset config at run time.
    """

    runtime: PartialRuntimeConfig | None = None
    split: SplitConfig | None = None
    output: PartialOutputConfig | None = None
    full_config_type: type[PartialDatasetConfig] = Field(
        default=PartialDatasetConfig, init=False, repr=False
    )

    @classmethod
    def from_inputs(
        cls,
        split_strategy: SplitStrategy | None = None,
        read_split: list[DatasetSplit] | None = None,
        jobs: int | Literal["auto"] | None = None,
        trajectory_schema: str | None = None,
        ratio: tuple[float, float, float] | None = None,
        gap: int | None = None,
        segments: int | None = None,
    ) -> RuntimeOverride:
        """Construct a runtime override from raw inputs.

        Inputs are based on raw CLI arguments that are not validated.

        !!! note "`PartialDatasetConfig` conversion"
            The `apply_to` method can be used to convert this `RuntimeOverride`
            into a `PartialDatasetConfig` that can be merged with the dataset's
            default config. This allows for applying overrides without needing
            to specify the full config structure.

        Parameters
        ----------
        split_strategy: SplitStrategy or None
            The strategy to use for splitting the dataset. If None, no override
            is applied.
        read_split: list of DatasetSplit or None
            The dataset splits to read from the dataset. Only applicable if
            split_strategy is "native". If None, no override is applied.
        jobs: int or Literal["auto"] or None
            The number of parallel jobs to use for loading and processing the
            dataset. If None, no override is applied. If set to "auto", the
            number of jobs will be set to the number of CPU cores.
        trajectory_schema: str or None
            The trajectory schema to use for the output trajectories. If None,
            no override is applied.
        ratio: tuple of three floats or None
            The ratio of train, validation, and test splits to use for splitting
            the dataset. Only applicable if split_strategy is "scene", "source",
            "time", or "shuffled-time". If None, no override is applied.
        gap: int or None
            The gap (in frames) to use between splits when using "scene", "source",
            "time", or "shuffled-time" split strategies. If None, no override is
            applied.
        segments: int or None
            The number of segments to split each scene into when using "scene" or
            "source" split strategies. If None, no override is applied.

        Returns
        -------
        RuntimeOverride
            A runtime override object containing the specified overrides.

        """
        split_data = {
            "strategy": split_strategy,
            "ratio": {"train": ratio[0], "val": ratio[1], "test": ratio[2]} if ratio else None,
            "gap": gap,
            "segments": segments,
            "splits": read_split,
        }
        split_data = {k: v for k, v in split_data.items() if v is not None}
        split_config = SplitConfig.model_validate(split_data) if "strategy" in split_data else None

        return cls(
            runtime=PartialRuntimeConfig(jobs=jobs) if jobs is not None else None,
            split=split_config,
            output=PartialOutputConfig(trajectory_schema=trajectory_schema)
            if trajectory_schema is not None
            else None,
        )
