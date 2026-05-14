"""Read-selection and assignment configuration models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, RootModel, model_validator

from dronalize.config.base import ResolvedConfig
from dronalize.core.categories import DatasetSplit  # noqa: TC001
from dronalize.core.errors import ConfigurationError


class SplitWeights(ResolvedConfig):
    """Weights used when routing data into train/val/test assignments."""

    train: float = Field(ge=0, default=0.0)
    """Weight assigned to the training split."""
    val: float = Field(ge=0, default=0.0)
    """Weight assigned to the validation split."""
    test: float = Field(ge=0, default=0.0)
    """Weight assigned to the test split."""

    @model_validator(mode="after")
    def _validate_sum(self) -> SplitWeights:
        """Validate that at least one split weight is positive."""
        total = self.train + self.val + self.test
        if total == 0:
            v = f"train={self.train}, val={self.val}, test={self.test}"
            msg = f"At least one split weight must be greater than 0, got {v}"
            raise ConfigurationError(msg)
        return self


class ReadAll(ResolvedConfig):
    """Read the full dataset input surface."""

    strategy: Literal["all"] = Field("all", repr=False, init=False)


class ReadNative(ResolvedConfig):
    """Read only selected dataset-native partitions."""

    strategy: Literal["native"] = Field("native", repr=False, init=False)
    splits: frozenset[DatasetSplit] | None = None
    """Native dataset partitions to read from the DatasetSource dataset."""


ReadUnion = Annotated[ReadAll | ReadNative, Field(discriminator="strategy")]


class ReadConfig(RootModel[ReadUnion]):
    """Read-selection configuration wrapper model."""

    root: ReadUnion


class NoAssign(ResolvedConfig):
    """No output split assignment."""

    strategy: Literal["none"] = Field("none", repr=False, init=False)


class PreserveNativeAssign(ResolvedConfig):
    """Preserve dataset-native split labels in output."""

    strategy: Literal["preserve-native"] = Field("preserve-native", repr=False, init=False)


class SceneAssign(ResolvedConfig):
    """Scene-based output split assignment."""

    strategy: Literal["scene"] = Field("scene", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


class SourceAssign(ResolvedConfig):
    """DatasetSource-based output split assignment."""

    strategy: Literal["source"] = Field("source", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


class TimeBlockAssign(ResolvedConfig):
    """Time-block output split assignment."""

    gap: int = Field(ge=0, default=0)
    strategy: Literal["time"] = Field("time", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


class ShuffledTimeBlockAssign(ResolvedConfig):
    """Shuffled time-block output split assignment."""

    segments: int = Field(ge=1)
    gap: int = Field(ge=0, default=0)
    strategy: Literal["shuffled-time"] = Field("shuffled-time", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)


AssignUnion = Annotated[
    NoAssign
    | PreserveNativeAssign
    | SceneAssign
    | SourceAssign
    | TimeBlockAssign
    | ShuffledTimeBlockAssign,
    Field(discriminator="strategy"),
]


class AssignConfig(RootModel[AssignUnion]):
    """Assignment configuration wrapper model."""

    root: AssignUnion
