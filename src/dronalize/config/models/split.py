"""Split strategy configuration models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, RootModel, model_validator

from dronalize.config.base import FullConfig
from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import ConfigurationError


class SplitWeights(FullConfig):
    """Weights used when routing data into train/val/test splits."""

    train: float = Field(ge=0, default=0.0)
    """Weight assigned to the training split."""
    val: float = Field(ge=0, default=0.0)
    """Weight assigned to the validation split."""
    test: float = Field(ge=0, default=0.0)
    """Weight assigned to the test split."""

    @model_validator(mode="after")
    def _validate_sum(self) -> SplitWeights:
        """Validate that the weights sum to 1."""
        total = self.train + self.val + self.test
        if total == 0:
            v = f"train={self.train}, val={self.val}, test={self.test}"
            msg = f"At least one split weight must be greater than 0, got {v}"
            raise ConfigurationError(msg)
        return self


class TimeSplitConfig(FullConfig):
    """Time-block split specification."""

    gap: int = Field(ge=0, default=0)
    """Number of scenes to leave between adjacent temporal split partitions."""
    strategy: Literal["time"] = Field("time", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)
    """Relative train/validation/test allocation for the time-based split."""


class ShuffledTimeSplitConfig(FullConfig):
    """Shuffled time-block split specification."""

    segments: int = Field(ge=1)
    """Number of time segments created before shuffling them across splits."""
    gap: int = Field(ge=0, default=0)
    """Number of scenes to leave between adjacent temporal split partitions."""
    strategy: Literal["shuffled-time"] = Field("shuffled-time", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)
    """Relative train/validation/test allocation for the shuffled time split."""


class SceneSplitConfig(FullConfig):
    """Scene-based split specification."""

    strategy: Literal["scene"] = Field("scene", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)
    """Relative train/validation/test allocation for random scene assignment."""


class SourceSplitConfig(FullConfig):
    """Source-based split specification."""

    strategy: Literal["source"] = Field("source", repr=False, init=False)
    ratio: SplitWeights = Field(default_factory=SplitWeights)
    """Relative train/validation/test allocation for source-level assignment."""


class NativeSplitConfig(FullConfig):
    """Dataset-defined split specification."""

    strategy: Literal["native"] = Field("native", repr=False, init=False)
    splits: frozenset[DatasetSplit] = Field(
        default=frozenset((DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)),
        max_length=3,
        min_length=1,
    )
    """Native dataset split labels to read from the source dataset."""


class NoSplitConfig(FullConfig):
    """No-split strategy specification."""

    strategy: Literal["none"] = Field("none", repr=False, init=False)


SplitConfigUnion = Annotated[
    TimeSplitConfig
    | ShuffledTimeSplitConfig
    | SceneSplitConfig
    | SourceSplitConfig
    | NativeSplitConfig
    | NoSplitConfig,
    Field(discriminator="strategy"),
]


class SplitConfig(RootModel[SplitConfigUnion]):
    """Split configuration wrapper model for discriminated union behavior."""

    root: SplitConfigUnion
    """Concrete split strategy configuration selected for the dataset."""
