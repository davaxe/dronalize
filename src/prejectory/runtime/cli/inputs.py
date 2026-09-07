"""Translate CLI flags into ordinary configuration patches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from prejectory.config.models import (
    AssignConfig,
    AssignStrategy,
    DatasetConfigPatch,
    OutputPatch,
    ReadConfig,
    ReadStrategy,
    RuntimePatch,
)
from prejectory.core.errors import ConfigurationError

if TYPE_CHECKING:
    from prejectory.core.categories import DatasetSplit


def _validate_read_inputs(
    *,
    read_strategy: ReadStrategy | None,
    read_split: list[DatasetSplit] | None,
) -> None:
    if read_split is None:
        return
    if read_strategy != "native":
        msg = "`read_split` requires `read_strategy='native'`."
        raise ConfigurationError(msg)


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


def config_overrides(
    read_strategy: ReadStrategy | None = None,
    read_split: list[DatasetSplit] | None = None,
    assign_strategy: AssignStrategy | None = None,
    jobs: int | Literal["auto"] | None = None,
    trajectory_schema: str | None = None,
    ratio: tuple[float, float, float] | None = None,
    gap: int | None = None,
    segments: int | None = None,
) -> DatasetConfigPatch:
    """Validate CLI options and return a dataset configuration patch."""
    _validate_read_inputs(read_strategy=read_strategy, read_split=read_split)
    _validate_assign_inputs(
        assign_strategy=assign_strategy,
        ratio=ratio,
        gap=gap,
        segments=segments,
    )
    read_data = {"strategy": read_strategy, "splits": read_split}
    read_data = {k: v for k, v in read_data.items() if v is not None}
    read_config = (
        TypeAdapter[ReadConfig](ReadConfig).validate_python(read_data)
        if "strategy" in read_data
        else None
    )
    assign_data = {
        "strategy": assign_strategy,
        "ratio": {"train": ratio[0], "val": ratio[1], "test": ratio[2]} if ratio else None,
        "gap": gap,
        "segments": segments,
    }
    assign_data = {k: v for k, v in assign_data.items() if v is not None}
    assign_config = (
        TypeAdapter[AssignConfig](AssignConfig).validate_python(assign_data)
        if "strategy" in assign_data
        else None
    )

    return DatasetConfigPatch(
        runtime=RuntimePatch(jobs=jobs) if jobs is not None else None,
        read=read_config,
        assign=assign_config,
        output=OutputPatch(trajectory_schema=trajectory_schema)
        if trajectory_schema is not None
        else None,
    )
