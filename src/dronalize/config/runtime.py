from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from dronalize.config.base import ConfigPatch
from dronalize.config.models.dataset import PartialDatasetConfig
from dronalize.config.models.output import PartialOutputConfig
from dronalize.config.models.runtime import PartialRuntimeConfig
from dronalize.config.models.split import AssignConfig, ReadConfig
from dronalize.core.errors import ConfigurationError

if TYPE_CHECKING:
    from dronalize.core.categories import DatasetSplit

ReadStrategy = Literal["all", "native"]
"""Supported runtime read-strategy override names accepted by the CLI layer.

The string values map directly to the concrete read config models in
[`dronalize.config.models.split`][].
"""

AssignStrategy = Literal["none", "preserve-native", "scene", "source", "time", "shuffled-time"]
"""Supported runtime assignment-strategy override names accepted by the CLI layer.

The string values map directly to the concrete assignment config models in
[`dronalize.config.models.split`][].
"""


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
